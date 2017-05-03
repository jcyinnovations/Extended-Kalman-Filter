#include <iostream>
#include "tools.h"
#include <math.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using namespace std;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    /**
     * Calculate the RMSE here.
     */

    VectorXd rmse(4);
    rmse << 0,0,0,0;

    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    //  * the estimation vector size should equal ground truth vector size
    if(estimations.size() != ground_truth.size()
            || estimations.size() == 0){
        cout << "Invalid estimation or ground_truth data" << endl;
        return rmse;
    }

    //accumulate squared residuals
    for(unsigned int i=0; i < estimations.size(); ++i){
        VectorXd residual = estimations[i] - ground_truth[i];
        residual = residual.array()*residual.array();
        rmse += residual;
    }

    rmse = rmse/estimations.size();
    rmse = rmse.array().sqrt();
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    /**
     * Calculate a Jacobian here.
     */
	MatrixXd Hj = MatrixXd::Zero(3,4);
	//recover state parameters
	double px = x_state(0);
	double py = x_state(1);
	double vx = x_state(2);
	double vy = x_state(3);

	//pre-compute a set of terms to avoid repeated calculation
	double f1 = pow(px,2) + pow(py,2);
	double f2 = sqrt(f1);
	double f3 = (f1*f2);

	//check division by zero
	if(fabs(f1) < 0.0001){
		cout << "CalculateJacobian () - Error - Division by Zero" << endl;
		return Hj;
	}

	//compute the Jacobian matrix
	Hj << (px/f2), (py/f2), 0, 0,
		  -(py/f1), (px/f1), 0, 0,
		  py*(vx*py - vy*px)/f3, px*(vy*px - vx*py)/f3, px/f2, py/f2;

	return Hj;
}

double Tools::NormalizePhi(double phi) {
    double p = phi;
    if (phi > M_PI) {
        p = phi - 2*M_PI;
    } else
    if (phi < -M_PI) {
        p = phi + 2*M_PI;
    }
    return p;
}
