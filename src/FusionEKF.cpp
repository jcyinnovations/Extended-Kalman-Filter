#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

    /**
     * Finish initializing the FusionEKF.
     * Set the process and measurement noises
     */
    ekf_ = KalmanFilter();

}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
    long noise_ax = 9;
    long noise_ay = 9;

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
     * Initialize the state ekf_.x_ with the first measurement.
     * Create the covariance matrix.
     * Remember: you'll need to convert radar from polar to cartesian coordinates.
     */
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;
    previous_timestamp_ = measurement_pack.timestamp_;
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ <<  1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1000, 0,
                0, 0, 0, 1000;
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ <<  1, 0, 1, 0,
                0, 1, 0, 1,
                0, 0, 1, 0,
                0, 0, 0, 1;
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ <<  1, 0, 1, 0,
                0, 1, 0, 1,
                1, 0, 1, 0,
                0, 1, 0, 1;
    ekf_.H_ = MatrixXd(2, 4);
    ekf_.H_ <<  1, 0, 0, 0,
                0, 1, 0, 0;
    //ekf_.Init(x_in, P_in, F_in, H_in, R_in, Q_in);

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        /**
         * Convert radar from polar to cartesian coordinates and initialize state.
         */
        ekf_.R_ = R_radar_;

        ekf_.x_(0) = measurement_pack.raw_measurements_(0) * cos(measurement_pack.raw_measurements_(1));
        ekf_.x_(1) = measurement_pack.raw_measurements_(0) * sin(measurement_pack.raw_measurements_(1));
        //Rough estimate of vx, vy using components of rho_dot
        ekf_.x_(2) = measurement_pack.raw_measurements_(2) * cos(measurement_pack.raw_measurements_(1));
        ekf_.x_(3) = measurement_pack.raw_measurements_(2) * sin(measurement_pack.raw_measurements_(1));
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
        /**
         * Initialize state.
         */
        ekf_.R_ = R_laser_;
        ekf_.x_(0) = measurement_pack.raw_measurements_(0);
        ekf_.x_(1) = measurement_pack.raw_measurements_(1);
   }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

    /**
   TODO:
    * Update the state transition matrix F according to the new elapsed time.
    - Time is measured in seconds.
    * Update the process noise covariance matrix.
    * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
    */
    double dt = (measurement_pack.timestamp_ - previous_timestamp_)/1000000.0;
    previous_timestamp_ = measurement_pack.timestamp_;
    double dt_2 = pow(dt,2);
	double dt_3 = dt_2*dt;
	double dt_4 = dt_3*dt;

    ekf_.F_ <<  1, 0, dt, 0,
                0, 1, 0, dt,
                0, 0, 1, 0,
                0, 0, 0, 1;
    //MatrixXd G = MatrixXd(4, 2);
    //ekf_.Q_ = G*Qu*Gt;
	ekf_.Q_ << dt_4*noise_ax/4, 0, dt_3*noise_ax/2, 0,
			   0, dt_4*noise_ay/4, 0, dt_3*noise_ay/2,
			   dt_3*noise_ax/2, 0, dt_2*noise_ax, 0,
			   0, dt_3*noise_ay/2, 0, dt_2*noise_ay;

    //VectorXd u = VectorXd(4);
    ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

   /**
    * Use the sensor type to perform the update step.
    * Update the state and covariance matrices.
    */

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
        VectorXd z = VectorXd(3);
        z << measurement_pack.raw_measurements_(0), measurement_pack.raw_measurements_(1), measurement_pack.raw_measurements_(2);
        ekf_.R_ = R_radar_;
        ekf_.UpdateEKF(z);
    } else {
    // Laser updates
        VectorXd z = VectorXd(2);
        ekf_.R_ = R_laser_;
        z << measurement_pack.raw_measurements_(0), measurement_pack.raw_measurements_(1);
        ekf_.Update(z);
    }

    // print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}
