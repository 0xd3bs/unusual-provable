# Udetect - Technical Implementation Details

This directory contains the Cairo implementation of the Principal Component Analysis (PCA) algorithm. Below are the key technical specifications and implementation details.

> **Note**: This implementation uses [Orion](https://github.com/gizatechxyz/orion) (currently in Public Archive), a framework developed by [Giza](https://github.com/gizatechxyz).

## Development Environment

- **Scarb Version**: 2.3.0
- **Cairo Version**: 2.3.0 (Required for Orion framework compatibility)
- **Sierra Version**: 1.3.0

## Execution Instructions

### Running the Model
To get the prediction result, run:
```bash
scarb cairo-run --available-gas 4000000000
```

### Running Tests
To execute unit tests, run:
```bash
scarb test
```

The quality of the prediction is evaluated by comparing the results with the Python version of this model. The test ensures that our Cairo implementation matches the original Python implementation's output.

## Technical Specifications

### Number Representation
- Uses Fixed Point 16x16 format for decimal number representation
- Implementation leverages Orion's `FP16x16` type for mathematical operations
- All calculations maintain precision through fixed-point arithmetic

### Key Components

- **Sorting Module**: Implements eigenvalue and eigenvector sorting
  - Located in `sorted.cairo`
  - Handles descending order sorting for eigenvalues
  - Manages corresponding eigenvector reordering

### Input/Output Format

- **Input**: 
  - Matrices are represented as Orion tensors
  - Data is preprocessed and standardized before PCA computation

- **Output**: 
  - Returns prediction as a decimal value (converted from FP16x16)
  - Example: 93 represents 93% variance explained by principal components

### Dependencies

- **[Orion](https://github.com/gizatechxyz/orion)**: Used for tensor operations and mathematical computations (Public Archive)

## Acknowledgments

We would like to express our sincere gratitude to:

- [Starknet Foundation](https://starknet.io/foundation/) for their invaluable support throughout the development of this project
- [Giza](https://github.com/gizatechxyz) for providing the Orion framework that made this implementation possible

## Testing

Tests are integrated into `lib.cairo` using Cairo's testing framework. The implementation is validated against Python notebook results to ensure accuracy.

## Usage Note

When interpreting results, note that the final output is automatically converted from fixed-point (FP16x16) to decimal format for easier understanding.
