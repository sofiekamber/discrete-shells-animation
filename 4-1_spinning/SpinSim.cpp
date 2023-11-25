#include <Eigen/SVD>
#include "SpinSim.h"

/////////////////////////////////////////
//////// EX 3 - PROBLEM 1 & 2 ///////////
/////////////////////////////////////////
bool SpinSim::advance() {
	Eigen::Vector3d w = p_body->getAngularVelocity();
	
	// TODO: update orientation
	switch (m_method) {
		case 0: {
			// matrix-based angular velocity
			Eigen::Matrix3d W;
			// skew-matrix (row-wise)
			W << 0, -w.z(), w.y(),
				w.z(), 0, -w.x(),
				-w.y(), w.x(), 0;

			Eigen::Matrix3d r = p_body->getRotationMatrix();
			r = r + m_dt * W * r;

			p_body->setRotation(r);
			break;
		}

		case 1: {
			// matrix-based angular velocity
			// orthogonalize rotation matrix to fix "explosion" issue
			// idea is to find the nearest orthogonal matrix by SVD
			Eigen::Matrix3d W;
			// skew-matrix (row-wise)
			W << 0, -w.z(), w.y(),
				w.z(), 0, -w.x(),
				-w.y(), w.x(), 0;

			Eigen::Matrix3d r = p_body->getRotationMatrix();
			r = r + m_dt * W * r;
			
			// orthogonalize rotation matrix to fix "explosion" issue
			// https://math.stackexchange.com/questions/3292034/normalizing-a-rotation-matrix
			// https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
			// idea is to find the nearest orthogonal matrix by SVD
			Eigen::JacobiSVD<Eigen::Matrix3d> svd(r, Eigen::ComputeFullV | Eigen::ComputeFullU);
			r = svd.matrixU() * svd.matrixV().transpose();

			p_body->setRotation(r);

			break;
		}

		case 2: {
			// quaternion-based angular velocity
			Eigen::Quaterniond wq;
			wq.w() = 0;
			wq.vec() = w;

			Eigen::Quaterniond q = p_body->getRotation();
			Eigen::Quaterniond dq = wq * q;
			Eigen::Quaterniond new_q;
			new_q.w() = q.w() + 0.5*m_dt*dq.w();
			new_q.vec() = q.vec() + 0.5*m_dt*dq.vec();
			p_body->setRotation(new_q.normalized());
			break;

		}
		default:{
			std::cerr << m_method << " is not a valid rotation method."
						<< std::endl;
		}
	}

	// advance time
	m_time += m_dt;
	m_step++;

	return false;
}