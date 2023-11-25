#include "SpringSim.h"

/////////////////////////////////////
//////// EX 1 ///////////
/////////////////////////////////////
bool SpringSim::advance() {
    // perform time integration with different integrators

	// useful functions: normalized(), norm()
    Eigen::Vector3d spring_dir =
        (m_spring.end - m_spring.start).normalized();
	float spring_norm = (m_spring.start - m_spring.end).norm();

	// compute external forces to get the acceleartion,
	// which is commonly used by the following methods

	// HINT: use p_cube, m_spring, m_dt, m_gravity
    Eigen::Vector3d f_int = -m_spring.stiffness * (spring_norm - m_spring.length) * spring_dir;
    Eigen::Vector3d f_damp = -m_spring.damping * p_cube->getLinearVelocity();
    Eigen::Vector3d f_ext = p_cube->getMass() * m_gravity;
    Eigen::Vector3d f = f_int + f_damp + f_ext;
    Eigen::Vector3d a = p_cube->getMassInv() * f;
	Eigen::Vector3d v = p_cube->getLinearVelocity();
	Eigen::Vector3d p = p_cube->getPosition();

	// note that it is required to update both m_spring.end and p_cube's position
    switch (m_method) {
        case 0:
        {
            // analytical solution
            float g = m_gravity.norm();
			float m = p_cube->getMass();
			float c1 = m*g / m_spring.stiffness;
			float alpha = -m_spring.damping / (2 * m);
			float beta = sqrt(4 * m_spring.stiffness*m - m_spring.damping*m_spring.damping) /
				(2 * m);
			float c2 = -c1*alpha / beta;
			
			float v_y = exp(alpha*m_time) *
				(c1*(alpha*cos(beta*m_time) - beta*sin(beta*m_time)) +
					c2*(alpha*sin(beta*m_time) + beta*cos(beta*m_time)));
			float p_y = exp(alpha*m_time) *
				(c1*cos(beta*m_time) + c2*sin(beta*m_time))
				- m_spring.length - m*g / m_spring.stiffness;

			p_cube->setLinearVelocity(Eigen::Vector3d(0, v_y, 0));
			p_cube->setPosition(Eigen::Vector3d(0, p_y, 0));
            break;
        }

        case 1:
        {
            // explicit euler
            p_cube->setPosition(p + m_dt*v);
            p_cube->setLinearVelocity(v + m_dt*a);
            break;
        }

        case 2:
        {
            // symplectic euler
            p_cube->setLinearVelocity(v + m_dt*a);
            p_cube->setPosition(p + m_dt*p_cube->getLinearVelocity());
            break;
        }
        case 3:
        {
            // explicit midpoint
            Eigen::Vector3d vm = v + 0.5*m_dt*a;
			p_cube->setPosition(p + m_dt*vm);
			Eigen::Vector3d pm = p + 0.5*m_dt*v;
			m_spring.end = pm;
			spring_dir = (m_spring.end - m_spring.start).normalized();
			spring_norm = (m_spring.start - m_spring.end).norm();
			f_int = -m_spring.stiffness*(spring_norm - m_spring.length) * spring_dir;
			f_damp = -m_spring.damping * vm;
			f_ext = p_cube->getMass() * m_gravity;
			f = f_int + f_damp + f_ext;
			a = f*p_cube->getMassInv();
			p_cube->setLinearVelocity(v + m_dt*a);
            break;
        }
        case 4:
        {
            // implicit euler
			float v_y = (p_cube->getMass()*v.y() - m_dt*
				(m_spring.stiffness*(p.y() + m_spring.length) - p_cube->getMass()*m_gravity.y())) /
				(m_spring.stiffness*m_dt*m_dt+m_spring.damping*m_dt+ p_cube->getMass());
			float p_y = p.y() + m_dt*v_y;
			p_cube->setLinearVelocity(Eigen::Vector3d(0, v_y, 0));
			p_cube->setPosition(Eigen::Vector3d(0, p_y, 0));
            break;
        }
        default:
            std::cerr << m_method << " is not a valid integrator method."
                        << std::endl;
    }

	// update spring end position
	m_spring.end = p_cube->getPosition();


    // advance m_time
    m_time += m_dt;
    m_step++;

    // log
    if ((m_step % m_log_frequency) == 0) {
        m_trajectories.back().push_back(p_cube->getPosition());
    }

    return false;
}
