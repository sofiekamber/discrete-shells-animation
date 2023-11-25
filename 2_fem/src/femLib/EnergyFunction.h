#pragma once 
#include <ObjectiveFunction.h>
#include "Element.h"

class EnergyFunction : public ObjectiveFunction {

public: EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
	EnergyFunction(){ }
	virtual ~EnergyFunction(void){ }
 
public:
	vector<const Element*> elements;

	bool APPLY_GRAVITY  = false;
	VectorXd f_ext;

	bool SOLVE_DYNAMICS = false;
	VectorXd get_a(const VectorXd &x) const {
		// (Optional) TODO: Compute current acceleration
		return (x - x_prev) / (h*h) - v_prev / h;
	}
	VectorXd x_prev;
	VectorXd v_prev;
	VectorXd M;
	double h; 
 
public:


	double evaluate(const VectorXd &x) const override {
		double totalEnergy = 0; 
		for (const auto &e : elements) {
			// TODO: Add up the energy for each element
			totalEnergy += e->energy(x);
		} 
		if (APPLY_GRAVITY) {
			// TODO: Add graviational potential energy
			totalEnergy -= x.dot(f_ext);
		}
		if (SOLVE_DYNAMICS) {
			// TODO: Add kinetic energy
			// Hint: You can compute the acceleration first in get_a()
			VectorXd a = get_a(x); 
			totalEnergy += .5 * pow(h, 2) * a.transpose() * M.asDiagonal() * a;
		} 
		return totalEnergy;
	}

	void addGradientTo(const VectorXd &x, VectorXd &grad) const  override {
		for (const auto &e : elements) {
			// TODO: Add the energy gradient for each element to the global vector
			VectorXd gradient_ = e->gradient(x);
			for (int i = 0; i < e->getNumNodes(); ++i) {
				grad.segment<2>(2 * e->getNodeIndex(i)) += gradient_.segment<2>(2 * i);
			}
		}
		if (APPLY_GRAVITY) { 
			// TODO: Add graviational potential energy gradient
			grad -= f_ext;
		}
		if (SOLVE_DYNAMICS) { 
			// TODO: Add kinetic energy gradient
			grad += M.asDiagonal() * get_a(x);
		}
	}

	void addHessianEntriesTo(const VectorXd &x, std::vector<Triplet<double>> &hessianEntries) const override {
		for (const auto &e : elements) {
			// TODO: Add the energy hessian for each element to the global hessian matrix
			// Hint: hessianEntries.push_back(Triplet<double>(row_index, col_index, entry_value));
			MatrixXd hessian_ = e->hessian(x);
			for (int i = 0; i < e->getNumNodes(); ++i) {
				for (int j = 0; j < e->getNumNodes(); ++j) { 
					for (int ii = 0; ii < 2; ++ii) {
						for (int jj = 0; jj < 2; ++jj) {
							hessianEntries.push_back(Triplet<double>(
								2 * e->getNodeIndex(i) + ii,
								2 * e->getNodeIndex(j) + jj,
								hessian_(2 * i + ii, 2 * j + jj)));
						}
					}
				}
			}
		}
		if (APPLY_GRAVITY) { 
			// TODO: Add graviational potential energy hessian
		}
		if (SOLVE_DYNAMICS) {
			// TODO: Add kinetic energy hessian
			for (int i = 0; i < x.size(); ++i) {
				hessianEntries.push_back(Triplet<double>(i, i, M[i] / pow(h, 2)));
			}
		}
	}

};

