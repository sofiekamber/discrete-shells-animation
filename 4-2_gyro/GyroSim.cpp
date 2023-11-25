#include "GyroSim.h"

/////////////////////////////////////////
//////// EX 3 - PROBLEM 3 & 4 ///////////
/////////////////////////////////////////
bool GyroSim::advance() {
	// update angular velocity
	Eigen::Vector3d w = p_body->getAngularVelocity();
	switch (m_method) {
	case 0: {
		// semi-implicit
		Eigen::Matrix3d Iinv = p_body->getInertiaInvWorld();
		Eigen::Vector3d res =
			m_dt * Iinv * (-w.cross(Iinv.inverse() * w));
		w += res;
		break;
	}
	case 1: {
		// solve gyroscopic
		Eigen::Quaterniond q = p_body->getRotation();
		Eigen::Matrix3d Ib = p_body->getInertia();
		double h = m_dt;

		// Convert to body coordinates
		Eigen::Vector3d omegab = q.inverse() * w;

		// Residual vector
		Eigen::Vector3d f = h * omegab.cross(Ib * omegab);

		// Jacobian
		Eigen::Matrix3d J = Ib + h * (skew(omegab) * Ib - skew(Ib * omegab));

		// Single Newton-Raphson update
		Eigen::Vector3d res = J.colPivHouseholderQr().solve(f);
		omegab = omegab - res;

		// Back to world coordinates
		w = q * omegab;
		break;
	}
	}

	p_body->setAngularVelocity(w);

	// update orientation
	Eigen::Quaterniond wq;
	wq.w() = 0;
	wq.vec() = w;

	Eigen::Quaterniond q = p_body->getRotation();
	Eigen::Quaterniond dq = wq * q;
	Eigen::Quaterniond new_q;
	new_q.w() = q.w() + 0.5 * m_dt * dq.w();
	new_q.vec() = q.vec() + 0.5 * m_dt * dq.vec();
	p_body->setRotation(new_q.normalized());

	// advance time
	m_time += m_dt;
	m_step++;

	return false;
}
