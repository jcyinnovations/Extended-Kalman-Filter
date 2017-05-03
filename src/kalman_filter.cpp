#include "kalman_filter.h"
#include <math.h>
#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

KalmanFilter::KalmanFilter() {
    I_ = MatrixXd::Identity(4, 4);
}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
    /**
     * predict the state
     */
    x_ = F_*x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_*P_*Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    /**
     * update the state by using Kalman Filter equations
     */
    VectorXd y = z - (H_*x_);
    MatrixXd Ht= H_.transpose();
    MatrixXd S = H_*P_*Ht + R_;
    MatrixXd Si= S.inverse();
    MatrixXd K = P_*Ht*Si;
    x_ = x_ + K*y;
    P_ = (I_ - K*H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    /**
     * update the state by using Extended Kalman Filter equations
     */
    Tools tools;
    MatrixXd Hj = tools.CalculateJacobian(x_);
    // Could not compute Jacobian if divide by zero so skip measurement
    if (Hj.sum() == 0.0)
        return;
    double px = x_(0);
    double py = x_(1);
    double vx = x_(2);
    double vy = x_(3);
    double hyp = sqrt( pow(px, 2) + pow(py,2) );
    VectorXd x_polar = VectorXd(3);

    x_polar << hyp, atan2(py,px), (px*vx + py*vy)/hyp;
    VectorXd y = z - x_polar;
    //Normalize the error angle to avoid drastic swings
    double phi = tools.NormalizePhi(y(1));
    y(1) = phi;
    //DEBUG: cout << "phi, " << y(1) << ", " << z(1) << ", " << x_polar(1) << endl;

    MatrixXd Ht= Hj.transpose();
    MatrixXd S = Hj*P_*Ht + R_;
    MatrixXd Si= S.inverse();
    MatrixXd K = P_*Ht*Si;
    MatrixXd I = MatrixXd::Identity(4, 4);
    x_ = x_ + K*y;
    P_ = (I - K*Hj) * P_;
}
