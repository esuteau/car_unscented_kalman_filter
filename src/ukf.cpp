#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 3.0;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.5;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

    // Debug Print Flag
    debug_print_ = false;

    // Start in uninitialized state.
    is_initialized_ = false;

    // Initialize time stamp to 0
    time_us_ = 0;

    // Set state dimension
    n_x_ = 5;

    // Set augmented dimension
    n_aug_ = 7;

    // Define spreading parameter
    lambda_ = 3 - n_aug_;

    // Number of sigma points
    nb_sigma_points_ = 2 * n_aug_ + 1;

    // Initialize Sigma Point Matrix
    Xsig_pred_ = MatrixXd(n_x_, nb_sigma_points_);

    // Initialize Weights
    weights_ = VectorXd(nb_sigma_points_);

    // Initialize NIS values for Radar and Laser
    NIS_radar_ = 0.0;
    NIS_laser_ = 0.0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
    /**
     TODO:

    Complete this function! Make sure you switch between lidar and radar
    measurements.
    */

    // Initialization
    if (!is_initialized_)
    {
        if (debug_print_)
            cout << "Initializing Filter Coefficients" << endl;

        // First measurement
        x_ = VectorXd(5);
        x_ << 1, 1, 1, 1, 0.2;

        if (debug_print_)
        cout << "X Initialized to: " << x_ << endl;

        // Initialize State covariance matrix P
        P_ << 0.2, 0, 0, 0, 0,
            0, 0.2, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;

        if (debug_print_)
            cout << "P Initialized to: " << P_ << endl;

        // Initialize Timestamp
        time_us_ = meas_package.timestamp_;
        
        if (debug_print_)
            cout << "Time Initialized to: " << time_us_ << endl;

        if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
        {
            // Convert radar from polar to cartesian coordinates and initialize state.
            float rho = meas_package.raw_measurements_[0];
            float phi = meas_package.raw_measurements_[1];
            float px = rho * cos(-phi);
            float py = rho * sin(-phi);
            x_ << px, py, meas_package.raw_measurements_[2], 0, 0;
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
        {
            // Initialize state.
            x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
        }

        // Initialize Weights
        weights_(0) = lambda_ / (lambda_ + n_aug_);;
        for (int i = 1; i < nb_sigma_points_; i++)
        {
            double weight = 0.5 / (n_aug_ + lambda_);
            weights_(i) = weight;
        }

        if (debug_print_)
            cout << "Weights Initialized to: " << weights_ << endl;

        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    // Compute the time elapsed between the current and previous measurements
    float dt = (meas_package.timestamp_ - time_us_) / 1000000.0; //dt - expressed in seconds
    time_us_ = meas_package.timestamp_;

    if (debug_print_)
        cout << "Delta T: " << time_us_ << endl;

    // Make Prediction
    Prediction(dt);

    // Measurement Update, if measurement sensor was selected
    if(use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
        UpdateLidar(meas_package);
    }

    if(use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        UpdateRadar(meas_package);
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t)
{
    /**
     TODO:

    Complete this function! Estimate the object's location. Modify the state
    vector, x_. Predict sigma points, the state, and the state covariance matrix.
    */

    if (debug_print_)
        cout << "Delta T: " << delta_t << endl;

    // Augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    // Augmented state covariance
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5, 5) = P_;
    P_aug(5, 5) = std_a_ * std_a_;
    P_aug(6, 6) = std_yawdd_ * std_yawdd_;

    if (debug_print_)
        cout << "Augmented State Covariange Matrix: " << P_aug << endl;

    // Square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    // Sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, nb_sigma_points_);
    Xsig_aug.col(0) = x_aug;
    for (int i = 0; i < n_aug_; i++)
    {
        Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
    }

    if (debug_print_)
        cout << "Augmented Sigma Point Matrix: " << Xsig_aug << endl;

    // Predict sigma points
    for (int i = 0; i < nb_sigma_points_; i++)
    {
        // Extract values for better readability
        double p_x = Xsig_aug(0, i);
        double p_y = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        // Predicted state values
        double px_p, py_p;

        // Avoid division by zero
        if (fabs(yawd) > 0.001)
        {
            px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        }
        else
        {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd * delta_t;
        double yawd_p = yawd;

        // Add noise
        px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p = v_p + nu_a * delta_t;
        yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_p = yawd_p + nu_yawdd * delta_t;

        // Write predicted sigma point into right column
        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = yaw_p;
        Xsig_pred_(4, i) = yawd_p;
    }

    if (debug_print_){
        cout << "Predicted Sigma Point Matrix Size: " << Xsig_pred_.size() << endl;
        cout << "Predicted Sigma Point Matrix: " << Xsig_pred_ << endl;
    }

    if (debug_print_){
        cout << "Weights Size: " << weights_.size() << endl;
        cout << "Weigths: " << weights_ << endl;
        //cout << "x_ Size: " << x_.size() << endl;
        //cout << "Sigma point matrix column size: " << Xsig_pred_.col(0).size() << endl;
    }

    // Calculate Predicted State Mean
    x_.fill(0.0);
    for (int i = 0; i < nb_sigma_points_; i++)
    {
        x_ = x_ + weights_(i) * Xsig_pred_.col(i);
    }

    if (debug_print_)
        cout << "Predicted State Mean: " << x_ << endl;

    // Calculate Predicted State Covariance Matrix
    P_.fill(0.0);
    for (int i = 0; i < nb_sigma_points_; i++)
    {
        // State difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        // Angle normalization (between -Pi and +Pi)
        x_diff(3) = atan2(sin(x_diff(3)), cos(x_diff(3)));

        // Update Covariance Matrix
        P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
    }

    if (debug_print_)
        cout << "Predicted Covariance Matrix: " << P_ << endl;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package)
{
    /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

    if (debug_print_){
        cout << "--------------------------" << endl;
        cout << "Measurement Update (LIDAR)" << endl;
    }

    // Set measurement dimension, lidar can measure px and py
    int n_z = 2;

    // Create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, nb_sigma_points_);

    // Transform Sigma Point to Radar Measurement Space
    for (int i = 0; i < nb_sigma_points_; i++)
    {
        // Extract values for better readibility
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);

        // Transform to measurement space
        Zsig(0, i) = p_x;
        Zsig(1, i) = p_y;
    }

    if (debug_print_)
        cout << "Sigma Points in LIDAR measurement space: " << Zsig << endl;

    // Calculate Mean Predicted Peasurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i = 0; i < nb_sigma_points_; i++)
    {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    if (debug_print_)
        cout << "Mean Predicted Measurement: " << z_pred << endl;

    // Calculate Measurement Covariance Matrix S
    MatrixXd S = MatrixXd(n_z, n_z);
    S.fill(0.0);
    for (int i = 0; i < nb_sigma_points_; i++)
    {
        // Residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // Update S
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    if (debug_print_)
        cout << "Measurement Covariance Matrix: " << S << endl;

    // Noise Covariance Matrix
    MatrixXd R = MatrixXd(n_z, n_z);
    R << std_laspx_ * std_laspx_, 0,
        0, std_laspy_ * std_laspy_;

    // Add Measurement Noise Covariance Matrix
    S = S + R;

    if (debug_print_)
        cout << "Measurement Covariance Matrix (With Noise): " << S << endl;

    // Create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);
    for (int i = 0; i < nb_sigma_points_; i++)
    {
        // Residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // State Difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        x_diff(3) = atan2(sin(x_diff(3)), cos(x_diff(3)));
        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    if (debug_print_)
        cout << "Cross Correlation (Tc): " << Tc << endl;

    // Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    if (debug_print_)
        cout << "Kalman Gain: " << K << endl;

    // Residual
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

    // Update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    // Calculate NIS
    NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;

    if (debug_print_){
        cout << "Mean State: " << x_ << endl;
        cout << "Covariance Matrix: " << x_ << endl;
        cout << "NIS: " << NIS_laser_ << endl;
        cout << "End of Measurement Update" << endl;
        cout << "-------------------------" << endl;
    }
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package)
{
    if (debug_print_){
        cout << "--------------------------" << endl;
        cout << "Measurement Update (RADAR)" << endl;
    }

    // Set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 3;

    // Create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, nb_sigma_points_);

    // Transform Sigma Point to Radar Measurement Space
    for (int i = 0; i < nb_sigma_points_; i++)
    {
        // Extract values for better readibility
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);
        double v1 = cos(yaw) * v;
        double v2 = sin(yaw) * v;

        // Transform to measurement space
        Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);                         //r
        Zsig(1, i) = atan2(p_y, p_x);                                     //phi
        Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); //r_dot
    }

    if (debug_print_)
        cout << "Sigma Points in RADAR measurement space: " << Zsig << endl;

    // Calculate Mean Predicted Peasurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i = 0; i < nb_sigma_points_; i++)
    {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    if (debug_print_)
        cout << "Mean Predicted Measurement: " << z_pred << endl;

    // Calculate Measurement Covariance Matrix S
    MatrixXd S = MatrixXd(n_z, n_z);
    S.fill(0.0);
    for (int i = 0; i < nb_sigma_points_; i++)
    {
        // Residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // Angle normalization (Between -Pi and +Pi)
        z_diff(1) = atan2(sin(z_diff(1)), cos(z_diff(1)));

        // Update S
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    if (debug_print_)
        cout << "Measurement Covariance Matrix: " << S << endl;

    // Noise Covariance Matrix
    MatrixXd R = MatrixXd(n_z, n_z);
    R << std_radr_ * std_radr_, 0, 0,
        0, std_radphi_ * std_radphi_, 0,
        0, 0, std_radrd_ * std_radrd_;

    // Add Measurement Noise Covariance Matrix
    S = S + R;

    if (debug_print_)
        cout << "Measurement Covariance Matrix (With Noise): " << S << endl;

    // Create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);
    for (int i = 0; i < nb_sigma_points_; i++)
    {
        // Residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // Angle normalization
        z_diff(1) = atan2(sin(z_diff(1)), cos(z_diff(1)));

        // State Difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        // Angle Normalization
        x_diff(3) = atan2(sin(x_diff(3)), cos(x_diff(3)));

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    if (debug_print_)
        cout << "Cross Correlation (Tc): " << Tc << endl;

    // Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    if (debug_print_)
        cout << "Kalman Gain: " << K << endl;

    // Residual
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

    // Angle Normalization
    z_diff(1) = atan2(sin(z_diff(1)), cos(z_diff(1)));

    // Update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    // Calculate NIS
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

    if (debug_print_){
        cout << "Mean State: " << x_ << endl;
        cout << "Covariance Matrix: " << x_ << endl;
        cout << "NIS: " << NIS_laser_ << endl;
        cout << "End of Measurement Update" << endl;
        cout << "-------------------------" << endl;
    }

    // Calculate Radar NIS
    // TODO
}
