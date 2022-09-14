#include <assert.h>
#include <casacore/casa/Arrays/Array.h>
#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/tables/Tables/ScalarColumn.h>
#include <omp.h>

#include <chrono>
#include <complex>
#include <cstdlib>  // size_t
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <thread>
#include <cmath>

#include "CUDA/Generic/Generic.h"
#include "common/KernelTypes.h"
#include "common/Proxy.h"
#include "idg-common.h"
#include "idg-util.h"  // Don't need if not using the test Data class
#include "npy.hpp"

#include "lender.hpp"
#include "calibration.cuh"

#define TEST false
#define PAR_CHAN 128 // number of channels to put into one grid
#define CHAN_THR 1 // number of IDG instances to make 
// [AT 8 IDG, ADDING PROXY IN START IS TOO MUCH MEMORY]

const float SPEED_OF_LIGHT = 299792458.0;
const float MAX_BL_M = 15392.2;

bool global_reading = true;

using casacore::IPosition;
using std::complex;
using std::string;

struct metadata {
  unsigned int nr_polarizations;
  unsigned int nr_rows;
  unsigned int nr_stations;
  unsigned int nr_baselines;
  unsigned int nr_timesteps;
  unsigned int nr_channels;
  float integration_time;
};

metadata getMetadata(const string &ms_path) {
  casacore::MeasurementSet ms(ms_path);
  casacore::ROArrayColumn<casacore::Complex> data_column(
      ms, casacore::MS::columnName(casacore::MSMainEnums::DATA)); // CORRECTED_DATA
  metadata meta{};
  meta.nr_polarizations = 4;
  meta.nr_rows = data_column.nrow();
  meta.nr_stations = ms.antenna().nrow();
  meta.nr_baselines = (meta.nr_stations * (meta.nr_stations - 1)) / 2;
  // assume there's no autocorrelation in the data
  // assert(meta.nr_rows % meta.nr_baselines == 0);
  // meta.nr_timesteps = meta.nr_rows / meta.nr_baselines;
  meta.nr_timesteps = 1;

  casacore::ROScalarColumn<double> exposure_col(
      ms, casacore::MS::columnName(casacore::MSMainEnums::EXPOSURE));
  meta.integration_time = static_cast<float>(exposure_col.get(0));

  casacore::ROScalarColumn<int> num_chan_col(
      ms.spectralWindow(), casacore::MSSpectralWindow::columnName(
                               casacore::MSSpectralWindowEnums::NUM_CHAN));
  meta.nr_channels = static_cast<unsigned int>(num_chan_col.get(0));
  std::clog << "integration_time = " << meta.integration_time << std::endl;
  std::clog << "nr_rows = " << meta.nr_rows << std::endl;
  std::clog << "nr_stations = " << meta.nr_stations << std::endl;
  std::clog << "nr_baselines = " << meta.nr_baselines << std::endl;
  std::clog << "nr_timesteps = " << meta.nr_timesteps << std::endl;
  std::clog << "nr_channels = " << meta.nr_channels << " (using) " << PAR_CHAN << std::endl;
  std::clog << "nr_polarizations = " << meta.nr_polarizations << std::endl;

  return meta;
}

void reorderData(const unsigned int nr_timesteps,
                 const unsigned int nr_rows,
                 const unsigned int nr_channels,
                 const casacore::Array<double> &uvw_rows,
                 const casacore::Array<complex<float>> &data_rows,
                 idg::Array2D<idg::UVW<float>> &uvw,
                 idg::Array4D<complex<float>> &visibilities,
                 idg::Array1D<std::pair<unsigned int, unsigned int>> &correlations) {
// TODO use one of the specialization of Array to iterate.
#pragma omp parallel for default(none) shared(visibilities, uvw, uvw_rows, data_rows)
  for (unsigned int t = 0; t < nr_timesteps; ++t) {
    for (unsigned int rw = 0; rw < nr_rows; ++rw) {
      unsigned int row_i = rw + t * nr_rows;

      idg::UVW<float> idg_uvw = {float(uvw_rows(IPosition(2, 0, row_i))),
                                  float(uvw_rows(IPosition(2, 1, row_i))),
                                  float(uvw_rows(IPosition(2, 2, row_i)))};
      uvw(rw, t) = idg_uvw;

      std::pair<unsigned int, unsigned int> curr_pair = correlations(row_i);
      if (curr_pair.first == curr_pair.second) {
        // set to 0
        for (unsigned int chan = 0; chan < PAR_CHAN; ++chan) {
          visibilities(rw, t, chan, 0) = 0;
          visibilities(rw, t, chan, 1) = 0;
          visibilities(rw, t, chan, 2) = 0;
          visibilities(rw, t, chan, 3) = 0;
        }
      } else {
        

        for (unsigned int chan = 0; chan < PAR_CHAN; ++chan) {
          visibilities(rw, t, chan, 0) = data_rows(IPosition(3, 0, chan, row_i));
          visibilities(rw, t, chan, 1) = data_rows(IPosition(3, 1, chan, row_i));
          visibilities(rw, t, chan, 2) = data_rows(IPosition(3, 2, chan, row_i));
          visibilities(rw, t, chan, 3) = data_rows(IPosition(3, 3, chan, row_i));
        }
      }
    }
  }
}

void getData(const string &ms_path, metadata meta,
             idg::Array2D<idg::UVW<float>> &uvw,
             idg::Array1D<float> &frequencies,
             idg::Array1D<std::pair<unsigned int, unsigned int>> &correlations,
             idg::Array4D<complex<float>> &visibilities) {
  casacore::MeasurementSet ms(ms_path);
  casacore::ROArrayColumn<casacore::Complex> data_column(
      ms, casacore::MS::columnName(casacore::MSMainEnums::DATA));
  casacore::ROArrayColumn<double> uvw_column(
      ms, casacore::MS::columnName(casacore::MSMainEnums::UVW));
  casacore::ROScalarColumn<int> ant1(
      ms, casacore::MS::columnName(casacore::MSMainEnums::ANTENNA1));
  casacore::ROScalarColumn<int> ant2(
      ms, casacore::MS::columnName(casacore::MSMainEnums::ANTENNA2));
  casacore::ROArrayColumn<double> freqs(
      ms.spectralWindow(), casacore::MSSpectralWindow::columnName(
                               casacore::MSSpectralWindowEnums::CHAN_FREQ));

  std::clog
      << "Reading baseline pairs and frequencies data from the measurement set."
      << std::endl;
  casacore::Array<double> src_freqs = freqs(0);
  assert(src_freqs.size() == meta.nr_channels);
  for (unsigned int i = 0; i < PAR_CHAN; ++i)
    frequencies(i) = float(src_freqs(IPosition(1, i)));
  std::clog << "done with reading frequencies." << std::endl;

  casacore::Slicer first_int_rows(IPosition(1, 0), IPosition(1, meta.nr_rows));
  casacore::Vector<int> ant1_vec = ant1.getColumnRange(first_int_rows);
  casacore::Vector<int> ant2_vec = ant2.getColumnRange(first_int_rows);
#pragma omp parallel for default(none) shared(baselines, ant1_vec, ant2_vec)
  for (unsigned int i = 0; i < meta.nr_rows; ++i) {
    std::pair<unsigned int, unsigned int> curr_pair = {ant1_vec(i),
                                                       ant2_vec(i)};
    correlations(i) = curr_pair;
  }

  std::chrono::_V2::system_clock::time_point start =
      std::chrono::high_resolution_clock::now();
  std::chrono::_V2::system_clock::time_point stop;
  std::chrono::seconds duration;

  std::clog << "Reading measurement set." << std::endl;

  /**
   * TODO store data on disk in different order.
   * I tried a few simple things: reading all baselines for given timestep then
   *reorder; reading one row at a time then reorder. casacore doesn't seem to
   *like it within omp parallel.
   **/

  const casacore::Array<complex<float>> data_rows = data_column.getColumn();
  const casacore::Array<double> uvw_rows = uvw_column.getColumn();
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
  std::clog << "Done reading measurement set in " << duration.count() << "s"
            << std::endl;
  start = std::chrono::high_resolution_clock::now();
  reorderData(meta.nr_timesteps, meta.nr_rows, meta.nr_channels, 
              uvw_rows, data_rows, uvw, visibilities, correlations);
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
  std::clog << "Reordered visibilities in " << duration.count() << "s"
            << std::endl;
}

float computeImageSize(unsigned long grid_size, float end_frequency) {
  const float grid_padding = 1.20;
  grid_size /= (2 * grid_padding);
  return grid_size / MAX_BL_M * (SPEED_OF_LIGHT / end_frequency);
}


void ms_fill_thread(std::shared_ptr<library<std::complex<float>>> r3, 
             const string& ms_path, metadata meta,
             std::shared_ptr<library<float>> r_uvw,
             idg::Array1D<float> &frequencies,
             idg::Array1D<std::pair<unsigned int, unsigned int>> &correlations) {

  idg::Array4D<complex<float>> main_vis = idg::Array4D<complex<float>>(meta.nr_rows, meta.nr_timesteps, PAR_CHAN, meta.nr_polarizations);
  
  auto uvw_b = r_uvw->fill();
  idg::Array2D<idg::UVW<float>> uvw((idg::UVW<float>*) uvw_b.get(), meta.nr_rows, meta.nr_timesteps);

  std::clog << ">>> Reading data" << std::endl;
  getData(ms_path, meta, uvw, frequencies, correlations, main_vis);

  r_uvw->queue(uvw_b);

  // start timing, calculate how much
  std::chrono::_V2::steady_clock::time_point start;
  std::chrono::_V2::steady_clock::time_point stop;
  std::chrono::milliseconds duration;
  // while(global_reading) {
    start = std::chrono::steady_clock::now();
    std::clog << ">>> Copying main data" << std::endl;
    auto vis = r3->fill();
    idg::Array4D<complex<float>> visibilities(vis.get(), meta.nr_rows, meta.nr_timesteps, PAR_CHAN, meta.nr_polarizations);
    memcpy(visibilities.data(), main_vis.data(), main_vis.bytes());
    r3->queue(vis);
    stop = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::clog << "Sent " << main_vis.bytes() << " in " << duration.count() << " milliseconds. " <<
      ((float)main_vis.bytes()/1e9)/((float)duration.count()/1000.0f) << " GB/s"<< std::endl;
  // }

  // global_reading = false;
}

void grid_operate_thread(std::shared_ptr<library<std::complex<float>>> r3,
             std::shared_ptr<library<bool>> flag_mask,
             metadata meta,
             std::shared_ptr<library<float>> r_uvw,
             idg::Array1D<float> &frequencies,
             idg::Array1D<std::pair<unsigned int, unsigned int>> &correlations) { // proxy
  
  std::clog << ">>> Initialize IDG proxy." << std::endl;
  idg::proxy::cuda::Generic proxy;

  float end_frequency = frequencies(frequencies.size() - 1);
  std::clog << "end frequency = " << end_frequency << std::endl;

  unsigned int grid_size = 8192;
  float image_size = computeImageSize(grid_size, end_frequency);
  float cell_size = image_size / grid_size;
  std::clog << "grid_size = " << grid_size << std::endl;
  std::clog << "image_size = " << image_size << " (radians?)" << std::endl;
  std::clog << "pixel_size (idg's cell_size) = " << cell_size << " (radians?)" << std::endl;

  // A-terms
  const unsigned int nr_timeslots = 1;  // timeslot for a-term
  const unsigned int subgrid_size = 32;
  const unsigned int kernel_size = 13;
  idg::Array4D<idg::Matrix2x2<complex<float>>> aterms = idg::get_example_aterms(
      proxy, nr_timeslots, meta.nr_stations, subgrid_size, subgrid_size);
  idg::Array1D<unsigned int> aterms_offsets =
      idg::get_example_aterms_offsets(proxy, nr_timeslots, meta.nr_timesteps);

  idg::Array2D<float> spread =
      idg::get_example_spheroidal(proxy, subgrid_size, subgrid_size);
  idg::Array1D<float> shift = idg::get_zero_shift();

  std::shared_ptr<idg::Grid> grid =
      proxy.allocate_grid(1, meta.nr_polarizations, grid_size, grid_size);
  proxy.set_grid(grid);
  // no w-tiling, i.e. not using w_step

  proxy.init_cache(subgrid_size, cell_size, 0.0, shift);

  auto uvw_b = r_uvw->operate();
  idg::Array2D<idg::UVW<float>> uvw((idg::UVW<float>*) uvw_b.get(), meta.nr_rows, meta.nr_timesteps);

  // Create reference array (vector)
  std::vector<long unsigned> im_shape {4, grid_size, grid_size};
  auto ref_iquv = proxy.allocate_array3d<double>(4, grid_size, grid_size);
  std::vector<double> temp_data = std::vector<double>(ref_iquv.data(),
                            ref_iquv.data() + ref_iquv.size());
  bool t = false;
  npy::LoadArrayFromNumpy("reference.npy", im_shape, t, temp_data);

  // Create plan
  std::clog << ">>> Creating plan" << std::endl;
  idg::Plan::Options options;
  options.plan_strict = true;
  const std::unique_ptr<idg::Plan> plan = proxy.make_plan(
      kernel_size, frequencies, uvw, correlations, aterms_offsets, options);
  std::clog << std::endl;

  // while (global_reading) {
    buffer_ptr vis = r3->operate();

    // Do flagging
    auto flags = flag_mask->fill();
    memset(flags.get(), 0x00, sizeof(bool)*flags.size);

    // Application in GPU (just switching #s for nchan and nbaseline for now)
    std::clog << ">>> Running flagging" << std::endl;
    call_flag_mask_kernel(meta.nr_rows, PAR_CHAN, meta.nr_polarizations, flags.get(), (float*) vis.get());

    idg::Array4D<complex<float>> visibilities(vis.get(), meta.nr_rows, 
                    meta.nr_timesteps, PAR_CHAN, meta.nr_polarizations);

    std::chrono::_V2::steady_clock::time_point start =
      std::chrono::steady_clock::now();
    std::chrono::_V2::steady_clock::time_point stop;
    std::chrono::milliseconds duration;

    std::clog << ">>> Run gridding" << std::endl;
    proxy.gridding(*plan, frequencies, visibilities, uvw, correlations, aterms,
                  aterms_offsets, spread);
    proxy.get_final_grid();
    stop = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::clog << " >>> thread " << std::this_thread::get_id  << ": Gridding in " << duration.count() << "ms"<< std::endl;

    std::clog << "Run FFT" << std::endl;
    proxy.transform(idg::FourierDomainToImageDomain);
    auto image_corr = proxy.get_final_grid();
    auto image_iquv = proxy.allocate_array3d<double>(4, grid_size, grid_size);
    for (unsigned int i = 0; i < grid_size; ++i) {
      for (unsigned int j = 0; j < grid_size; ++j) {
        image_iquv(0, i, j) =
            static_cast<double>(0.5 * ((*image_corr)(0, 0, i, j).real() +
                                      (*image_corr)(0, 3, i, j).real()));
        image_iquv(1, i, j) =
            static_cast<double>(0.5 * ((*image_corr)(0, 0, i, j).real() -
                                      (*image_corr)(0, 3, i, j).real()));
        image_iquv(2, i, j) =
            static_cast<double>(0.5 * ((*image_corr)(0, 1, i, j).real() -
                                      (*image_corr)(0, 2, i, j).real()));
        image_iquv(3, i, j) =
            static_cast<double>(0.5 * (-(*image_corr)(0, 1, i, j).imag() +
                                      (*image_corr)(0, 2, i, j).imag()));
      }
    }

    if (TEST) {
      int total_incorrect = 0;
      for (size_t z = 0; z < ref_iquv.get_z_dim(); ++z) {
        for (size_t y = 0; y < ref_iquv.get_y_dim(); ++y) {
          for (size_t x = 0; x < ref_iquv.get_x_dim(); ++x) {
            if (std::fabs(ref_iquv(z, y, x) - image_iquv(z, y, x)) > 0.01) {
              ++total_incorrect;
            }
          }
        }
      }
      std::cout << "Testing image result. " << total_incorrect << " incorrect values." << std::endl;
    }

    const long unsigned imshape[] = {4, grid_size, grid_size};
    npy::SaveArrayAsNumpy(
        "image.npy", false, 3, imshape,
        std::vector<double>(image_iquv.data(),
                            image_iquv.data() + image_iquv.size()));
  // }
}


int main(int argc, char *argv[]) {
  string ms_path = "/fastpool/data/LWA_calibrated.ms";
  // string ms_path = "/fastpool/data/20210226M-1350MHz-1chan-1int-ground-truth.ms";

  metadata meta = getMetadata(ms_path);

  std::cout
        << "offset of u = " << offsetof(idg::UVW<float>, u) << '\n'
        << "offset of v = " << offsetof(idg::UVW<float>, v) << '\n'
        << "offset of w = " << offsetof(idg::UVW<float>, w) << '\n';

  std::clog << ">>> Allocating metadata arrays" << std::endl;
  std::vector<size_t> shape_freq {PAR_CHAN};
  library<float> r_freq = library<float>(shape_freq, 1);
  auto freq = r_freq.fill();
  std::vector<size_t> shape_correlations {meta.nr_rows};
  library<std::pair<unsigned int, unsigned int>> r_correlations = 
          library<std::pair<unsigned int, unsigned int>>(shape_correlations, 1);
  auto corrs = r_correlations.fill();

  idg::Array1D<float> frequencies(freq.get(), PAR_CHAN);
  idg::Array1D<std::pair<unsigned int, unsigned int>> correlations(corrs.get(),
          meta.nr_rows);

  std::vector<size_t> shape_uvw {meta.nr_rows, meta.nr_timesteps, 3};
  std::shared_ptr<library<float>> r_uvw = 
          std::make_shared<library<float>>(shape_uvw, 1);

  std::clog << ">>> Allocating vis" << std::endl;

  std::vector<size_t> shape {meta.nr_rows, meta.nr_timesteps, PAR_CHAN, meta.nr_polarizations};
  std::shared_ptr<library<complex<float>>> r3 = 
          std::make_shared<library<complex<float>>>(shape, CHAN_THR);

  std::shared_ptr<library<bool>> flag_mask = 
          std::make_shared<library<bool>>(shape, 1);

  std::thread measurement (ms_fill_thread, r3, ms_path, meta, r_uvw, 
          std::ref(frequencies), std::ref(correlations));

  std::vector<std::thread> threads(CHAN_THR);
  for (auto& i : threads) {
      i = std::thread(grid_operate_thread, r3, flag_mask, meta, r_uvw, 
                std::ref(frequencies), std::ref(correlations));
  }

  measurement.join();
  for (auto& i : threads) {
      i.join();
  }
}
