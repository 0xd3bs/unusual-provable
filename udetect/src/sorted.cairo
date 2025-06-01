use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{core::{Tensor, TensorTrait}};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16, FixedTrait};
use core::clone::Clone;

/// Module for sorting operations on tensors
/// This module provides functionality for sorting eigenvalues and eigenvectors

/// Helper function to find the maximum value and its index in an array
/// # Arguments
/// * `arr` - Reference to the array to search in
/// * `used` - Reference to boolean array indicating which elements have been used
/// * `len` - Length of the array
/// # Returns
/// * Tuple containing the maximum value and its index
fn find_max(arr: @Array<FP16x16>, used: @Array<bool>, len: usize) -> (FP16x16, usize) {
    let mut max_val = FP16x16 { mag: 0_u32, sign: false }; // Initialize with lowest possible value
    let mut max_idx = 0;
    let mut found = false;
    let mut i = 0;

    loop {
        if i >= len {
            break;
        };
        if !(*used.at(i)) && *arr.at(i) > max_val {
            max_val = *arr.at(i);
            max_idx = i;
        };
        i += 1;
    };

    (max_val, max_idx)
}

/// Helper function to update the array of used elements
/// # Arguments
/// * `used` - Reference to boolean array indicating which elements have been used
/// * `idx` - Index of the element to mark as used
/// * `len` - Length of the array
/// # Returns
/// * New array with the specified index marked as used
fn update_used(used: @Array<bool>, idx: usize, len: usize) -> Array<bool> {
    let mut new_used = ArrayTrait::new();
    let mut i = 0;

    loop {
        if i >= len {
            break;
        };
        if i == idx {
            new_used.append(true);
        } else {
            new_used.append(*used.at(i));
        };
        i += 1;
    };

    new_used
}

/// Sorts eigenvalues in descending order and returns their indices
/// # Arguments
/// * `evalu` - Reference to tensor containing eigenvalues
/// # Returns
/// * Tuple containing:
///   - Sorted eigenvalues as a tensor
///   - Span of indices indicating the original positions
fn evalu_sort(evalu: @Tensor<FP16x16>) -> (Tensor<FP16x16>, Span<usize>) {
    let mut shape = ArrayTrait::new();
    shape.append(3);

    // Extract data from the eigenvalues tensor
    let mut original_data = ArrayTrait::new();
    let mut i: usize = 0;
    loop {
        if i == 3_usize {
            break;
        }
        let value = (*evalu).at(indices: array![i].span());
        original_data.append(value);
        i += 1;
    };

    // Initialize tracking array for used elements
    let mut used = ArrayTrait::new();
    used.append(false);
    used.append(false);
    used.append(false);

    // Sort in descending order and track indices
    let mut sorted_data = ArrayTrait::new();
    let mut sorted_indices = ArrayTrait::new();
    let len = original_data.len();
    let mut remaining = len;

    loop {
        if remaining == 0 {
            break;
        }
        let (max_val, max_idx) = find_max(@original_data, @used, len);
        sorted_data.append(max_val);
        sorted_indices.append(max_idx);
        used = update_used(@used, max_idx, len);
        remaining -= 1;
    };

    let tensor = TensorTrait::<FP16x16>::new(shape.span(), sorted_data.span());
    return (tensor, sorted_indices.span());
}

/// Reorders eigenvectors based on sorted eigenvalue indices
/// # Arguments
/// * `evec` - Reference to tensor containing eigenvectors
/// * `indices` - Span of indices from sorted eigenvalues
/// # Returns
/// * Tensor containing reordered eigenvectors
fn evec_sort(evec: @Tensor<FP16x16>, indices: Span<usize>) -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    let mut col_idx = 0;
    loop {
        if col_idx == 3 {
            break;
        };

        // Get the correct column index from sorted indices
        let source_col = *indices.at(col_idx);

        // Copy the entire column from source to destination
        let mut row = 0;
        loop {
            if row == 3 {
                break;
            };
            let value = (*evec).at(indices: array![row, source_col].span());
            data.append(value);
            row += 1;
        };

        col_idx += 1;
    };

    let tensor = TensorTrait::<FP16x16>::new(shape.span(), data.span());
    return tensor;
}
