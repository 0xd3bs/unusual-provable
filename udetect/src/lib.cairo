mod generated;
mod helper;
mod sorted;

use debug::PrintTrait;
use traits::TryInto;
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{Tensor, TensorTrait};
use orion::numbers::fixed_point::{core::{FixedTrait}};

use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorDiv, FP16x16TensorSub};

use orion::numbers::fixed_point::implementations::fp16x16::core::{
    FP16x16, FP16x16Impl, FP16x16Add, FP16x16AddEq, FP16x16Sub, FP16x16Mul, FP16x16MulEq,
    FP16x16TryIntoU128, FP16x16PartialEq, FP16x16PartialOrd, FP16x16SubEq, FP16x16Neg, FP16x16Div,
    FP16x16IntoFelt252, FP16x16Print, ONE
};

use udetect::{
    helper::{
        EigenValues, extract_diagonal, eye, find_max_off_diag, jacobi_eigensystem,
        update_eigen_values, check_unit_diagonal_tensor, div_by_scalar, div_by_fp
    }
};

use udetect::{generated::{X_std::X_std, X::X, y::y}, sorted::{evalu_sort, evec_sort}};

fn main() -> u32 {
    let tol = FixedTrait::<FP16x16>::new(655, false); // 655 is 0.01 = 1e-2
    let max_iter = 500_usize;

    let X_std = X_std();
    let X = X();
    let y = y();

    let mut n: usize = *((X_std).shape.at(0)) - 1;
    let size = *(X_std.shape.at(1));

    let X_std_transpose = X_std.transpose(axes: array![1, 0].span());
    let mut cov_matrix = div_by_scalar(@(X_std_transpose.matmul(@X_std)), n);

    let mut stddevs = extract_diagonal(@cov_matrix).sqrt();

    let mut stddevs_left = stddevs.reshape(array![size, 1].span());
    let mut stddevs_right = stddevs.reshape(array![1, size].span());
    let corr_matrix = cov_matrix / stddevs_left.matmul(@stddevs_right);

    let (evalu, evec) = jacobi_eigensystem(a: corr_matrix, tol: tol, max_iter: max_iter);

    let (evalu_sorted, evalu_indices) = evalu_sort(@evalu);
    let evec_sorted = evec_sort(@evec, evalu_indices);

    let loadings = evec_sorted;

    let principal_component = X_std.matmul(@loadings);

    n = *((principal_component).shape.at(0)) - 1;

    let principal_component_transpose = principal_component.transpose(axes: array![1, 0].span());

    let cov_new = div_by_scalar(@(principal_component_transpose.matmul(@principal_component)), n);

    let stddevs = extract_diagonal(@cov_new).sqrt();
    let stddevs_left = stddevs.reshape(array![size, 1].span());
    let stddevs_right = stddevs.reshape(array![1, size].span());
    let corr_new = cov_new / stddevs_left.matmul(@stddevs_right);

    let new_corr = (@corr_new.abs()).round();

    let evalu_cumsum = evalu.cumsum(0, Option::None(()), Option::None(()));
    let sum = evalu_cumsum.data.at(evalu_cumsum.data.len() - 1);
    let evalu_div_sum = div_by_fp(@evalu, *sum);
    let pc = (*evalu_div_sum.data.at(0) + *evalu_div_sum.data.at(1))
        * FixedTrait::<FP16x16>::new_unscaled(100, false);

    // Converts to a decimal value from fp16x16
    // ONE is a constant value representing 1 in fp16x16
    (FixedTrait::round(pc).mag / ONE)
}

#[cfg(test)]
mod tests {
    use super::main;
    #[test]
    #[available_gas(99999999999999999)]
    fn udetect_test() {
        let mut result: u32 = main();

        // The obtained prediction is compared with the Python result from the notebook (97%)
        assert(result >= 90_u32, 'no match with notebook version');
    }
}
