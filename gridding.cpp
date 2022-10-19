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
const float MAX_BL_M = 2365.8; // max baseline for lwa

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
      ms, casacore::MS::columnName(casacore::MSMainEnums::CORRECTED_DATA)); // CORRECTED_DATA
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
                 const casacore::Array<bool> &flag_rows,
                 const casacore::Array<float> &weight_rows,
                 idg::Array2D<idg::UVW<float>> &uvw,
                 idg::Array4D<complex<float>> &visibilities,
                 idg::Array4D<bool> &flags,
                 idg::Array4D<float> &weights,
                 idg::Array1D<std::pair<unsigned int, unsigned int>> &correlations) {
// TODO use one of the specialization of Array to iterate.
#pragma omp parallel for default(none) shared(visibilities, uvw, flags, flag_rows, uvw_rows, data_rows)
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

          flags(rw, t, chan, 0) = 0;
          flags(rw, t, chan, 1) = 0;
          flags(rw, t, chan, 2) = 0;
          flags(rw, t, chan, 3) = 0;
        }
      } else {
        

        for (unsigned int chan = 0; chan < PAR_CHAN; ++chan) {
          visibilities(rw, t, chan, 0) = data_rows(IPosition(3, 0, chan, row_i));
          visibilities(rw, t, chan, 1) = data_rows(IPosition(3, 1, chan, row_i));
          visibilities(rw, t, chan, 2) = data_rows(IPosition(3, 2, chan, row_i));
          visibilities(rw, t, chan, 3) = data_rows(IPosition(3, 3, chan, row_i));

          flags(rw, t, chan, 0) = flag_rows(IPosition(3, 0, chan, row_i));
          flags(rw, t, chan, 1) = flag_rows(IPosition(3, 1, chan, row_i));
          flags(rw, t, chan, 2) = flag_rows(IPosition(3, 2, chan, row_i));
          flags(rw, t, chan, 3) = flag_rows(IPosition(3, 3, chan, row_i));

          weights(rw, t, chan, 0) = weight_rows(IPosition(3, 0, chan, row_i));
          weights(rw, t, chan, 1) = weight_rows(IPosition(3, 1, chan, row_i));
          weights(rw, t, chan, 2) = weight_rows(IPosition(3, 2, chan, row_i));
          weights(rw, t, chan, 3) = weight_rows(IPosition(3, 3, chan, row_i));
        }
      }
    }
  }
}

void getData(const string &ms_path, metadata meta,
             idg::Array2D<idg::UVW<float>> &uvw,
             idg::Array4D<bool> &flags,
             idg::Array4D<float> &weights,
             idg::Array1D<float> &frequencies,
             idg::Array1D<std::pair<unsigned int, unsigned int>> &correlations,
             idg::Array4D<complex<float>> &visibilities) {
  casacore::MeasurementSet ms(ms_path);
  casacore::ROArrayColumn<casacore::Complex> data_column(
      ms, casacore::MS::columnName(casacore::MSMainEnums::CORRECTED_DATA));
  casacore::ROArrayColumn<double> uvw_column(
      ms, casacore::MS::columnName(casacore::MSMainEnums::UVW));
  casacore::ROScalarColumn<int> ant1(
      ms, casacore::MS::columnName(casacore::MSMainEnums::ANTENNA1));
  casacore::ROScalarColumn<int> ant2(
      ms, casacore::MS::columnName(casacore::MSMainEnums::ANTENNA2));
  casacore::ROArrayColumn<double> freqs(
      ms.spectralWindow(), casacore::MSSpectralWindow::columnName(
                               casacore::MSSpectralWindowEnums::CHAN_FREQ));
  casacore::ROArrayColumn<bool> flag_column(
      ms, casacore::MS::columnName(casacore::MSMainEnums::FLAG));
  casacore::ROArrayColumn<float> weight_column(
      ms, casacore::MS::columnName(casacore::MSMainEnums::WEIGHT));

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
  const casacore::Array<bool> flag_rows = flag_column.getColumn();
  const casacore::Array<float> weight_rows = weight_column.getColumn();
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
  std::clog << "Done reading measurement set in " << duration.count() << "s"
            << std::endl;
  start = std::chrono::high_resolution_clock::now();
  reorderData(meta.nr_timesteps, meta.nr_rows, meta.nr_channels, 
              uvw_rows, data_rows, flag_rows, weight_rows, uvw, visibilities, 
              flags, weights, correlations);
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


void ms_fill_thread(std::shared_ptr<library<std::complex<float>>> r3, const int argc,
             char *argv[], metadata meta,
             std::shared_ptr<library<float>> r_uvw,
             std::shared_ptr<library<bool>> r_flag,
             std::shared_ptr<library<float>> r_weight,
             idg::Array1D<float> &frequencies,
             idg::Array1D<std::pair<unsigned int, unsigned int>> &correlations) {
  
  
  std::chrono::_V2::steady_clock::time_point start;
  std::chrono::_V2::steady_clock::time_point stop;
  std::chrono::milliseconds duration;
  
  // start loop through each arg
  string ms_path;
  for (int m = 1; m < argc; m++) {
    ms_path = argv[m];

    std::clog << ">>> Reading data" << std::endl;
    // start timing, calculate how much
    start = std::chrono::steady_clock::now();
    
    auto vis = r3->fill();
    idg::Array4D<complex<float>> visibilities(vis.get(), meta.nr_rows, meta.nr_timesteps, PAR_CHAN, meta.nr_polarizations);
      
    auto uvw_b = r_uvw->fill();
    idg::Array2D<idg::UVW<float>> uvw((idg::UVW<float>*) uvw_b.get(), meta.nr_rows, meta.nr_timesteps);

    auto flag_b = r_flag->fill();
    idg::Array4D<bool> flags((bool*) flag_b.get(), meta.nr_rows, meta.nr_timesteps, PAR_CHAN, meta.nr_polarizations);

    auto weight_b = r_weight->fill();
    idg::Array4D<float> weights(weight_b.get(), meta.nr_rows, meta.nr_timesteps, PAR_CHAN, meta.nr_polarizations);

    getData(ms_path, meta, uvw, flags, weights, frequencies, correlations, visibilities);

    r_uvw->queue(uvw_b);
    r_flag->queue(flag_b);
    r_weight->queue(weight_b);
    r3->queue(vis);

    stop = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::clog << "Total data read for " << ms_path << " in " << duration.count() << " milliseconds. " <<std::endl;
  }
}

using namespace std::complex_literals;
void fill_jones(std::shared_ptr<library<std::complex<float>>> jones_lib, metadata meta) {
  casacore::Table ms("/fastpool/data/test_bandpass.bcal");
  casacore::ROArrayColumn<casacore::Complex> data_column(
      ms, "CPARAM");
  const casacore::Array<complex<float>> data_rows = data_column.getColumn();
  std::clog << "data_rows " << data_rows.shape() << std::endl;
  
  auto jones = jones_lib->fill();

  std::clog << ">>> Guessing Jones Dimensions as " << meta.nr_stations << " stations and channels " << PAR_CHAN << std::endl;
  for (unsigned int st = 0; st < meta.nr_stations; ++st) {
    for (unsigned int chan = 0; chan < PAR_CHAN; ++chan) {
      jones[(st * PAR_CHAN * meta.nr_polarizations) + (chan * meta.nr_polarizations)] = data_rows(IPosition(3, 0, chan, st));
      jones[(st * PAR_CHAN * meta.nr_polarizations) + (chan * meta.nr_polarizations) + 1] = data_rows(IPosition(3, 0, chan, st));
      jones[(st * PAR_CHAN * meta.nr_polarizations) + (chan * meta.nr_polarizations) + 2] = data_rows(IPosition(3, 0, chan, st));
      jones[(st * PAR_CHAN * meta.nr_polarizations) + (chan * meta.nr_polarizations) + 3] = data_rows(IPosition(3, 0, chan, st));
    }
  }

  std::clog << ">>> Read table complete." << std::endl;

  // auto jones = jones_lib->fill();
  // // make into identity matrix
  // for (unsigned int a = 0; a < meta.nr_stations; a++) {
  //   for (unsigned int c = 0; c < PAR_CHAN; c++) {
  //     jones[(a * PAR_CHAN * meta.nr_polarizations) + (c * meta.nr_polarizations)] = 1. + 0i;
  //     jones[(a * PAR_CHAN * meta.nr_polarizations) + (c * meta.nr_polarizations) + 1] = 0. + 0i;
  //     jones[(a * PAR_CHAN * meta.nr_polarizations) + (c * meta.nr_polarizations) + 2] = 0. + 0i;
  //     jones[(a * PAR_CHAN * meta.nr_polarizations) + (c * meta.nr_polarizations) + 3] = 1. +0i;
  //   }
  // }

  jones_lib->queue(jones);
}

void grid_operate_thread(const int argc, std::shared_ptr<library<std::complex<float>>> r3,
             std::shared_ptr<library<bool>> flag_mask,
             std::shared_ptr<library<std::complex<float>>> jones_lib,
             std::shared_ptr<library<float>> weight_lib,
             metadata meta,
             std::shared_ptr<library<float>> r_uvw,
             idg::Array1D<float> &frequencies,
             idg::Array1D<std::pair<unsigned int, unsigned int>> &correlations) { // proxy
  
  // TODO: repeating the whole thing for now. In the future, want to only repeat necessary steps.
  for (int im = 1; im < argc; im++ ){
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

    buffer_ptr vis = r3->operate();

    // Do flagging
    auto flags = flag_mask->operate();
    // memset(flags.get(), 0x00, sizeof(bool)*flags.size);

    // Application in GPU (just switching #s for nchan and nbaseline for now)
    std::clog << ">>> Running flagging" << std::endl;
    call_flag_mask_kernel(meta.nr_rows, PAR_CHAN, meta.nr_polarizations, flags.get(), (float*) vis.get());

    auto jones = jones_lib->operate();
    call_jones_kernel(PAR_CHAN, meta.nr_rows, meta.nr_polarizations, meta.nr_stations, (float*) vis.get(), (int*) correlations.data(), (float*) jones.get());

    auto weight = weight_lib->operate();
    for (unsigned int i = 0; i < meta.nr_polarizations * meta.nr_rows * PAR_CHAN; i++) {
        vis[i] = vis[i] * weight[i];
    }

    // Create plan
    std::clog << ">>> Creating plan" << std::endl;
    idg::Plan::Options options;
    options.plan_strict = true;
    const std::unique_ptr<idg::Plan> plan = proxy.make_plan(
        kernel_size, frequencies, uvw, correlations, aterms_offsets, options);
    std::clog << std::endl;

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

    std::clog << ">>> Save Image" << std::endl;
    const long unsigned imshape[] = {4, grid_size, grid_size};
    npy::SaveArrayAsNumpy(
        std::to_string(im) + "_image.npy", false, 3, imshape,
        std::vector<double>(image_iquv.data(),
                            image_iquv.data() + image_iquv.size()));
  }
}


int main(int argc, char *argv[]) {

  if (argc < 2) {
    std::cout << "Expected file input" << std::endl;
    return 1;
  } else {
    for (int i = 0; i < argc; i++) {
      std::cout << argv[i] << std::endl;
    }
    std::cout << "argc " << argc << std::endl;
    string s = argv[1];
    if (s.find('*') < s.length()) {
      std::cout << "No matching files" << std::endl;
      return 1;
    }
  }

  metadata meta = getMetadata(argv[1]);

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
          std::make_shared<library<float>>(shape_uvw, argc);

  std::clog << ">>> Allocating vis" << std::endl;

  std::vector<size_t> shape {meta.nr_rows, meta.nr_timesteps, PAR_CHAN, meta.nr_polarizations};
  std::shared_ptr<library<complex<float>>> r3 = 
          std::make_shared<library<complex<float>>>(shape, argc);

  std::shared_ptr<library<bool>> flag_mask = 
          std::make_shared<library<bool>>(shape, argc);

  std::shared_ptr<library<float>> r_weight = 
          std::make_shared<library<float>>(shape, argc);

  std::vector<size_t> jshape {meta.nr_stations, PAR_CHAN, meta.nr_polarizations};
  std::shared_ptr<library<complex<float>>> jones_lib = 
          std::make_shared<library<complex<float>>>(shape, 1);

  std::thread measurement (ms_fill_thread, r3, argc, argv, meta, r_uvw, flag_mask, r_weight,
          std::ref(frequencies), std::ref(correlations));

  std::thread jones(fill_jones, jones_lib, meta);

  std::vector<std::thread> threads(CHAN_THR);
  for (auto& i : threads) {
      i = std::thread(grid_operate_thread, argc, r3, flag_mask, jones_lib, r_weight, meta, r_uvw, 
                std::ref(frequencies), std::ref(correlations));
  }

  jones.join();
  measurement.join();
  for (auto& i : threads) {
      i.join();
  }
}
