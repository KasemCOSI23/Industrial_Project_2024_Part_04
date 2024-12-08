# Industrial Project Part 4: PCA-Based Spectral Data Analysis

This project processes and analyzes spectral data using Principal Component Analysis (PCA). It supports multiple tasks such as `structure_oil`, `structure_charring`, `oil_charring`, and `cracking`. Tasks are configured via a `config2.yaml` file, enabling flexibility and reproducibility.

## Features

- **PCA Analysis**: Performs dimensionality reduction on spectral data and visualizes the explained variance and principal components.
- **Task-Based Execution**: Supports multiple tasks (e.g., `structure_oil`, `structure_charring`) specified in the configuration file.
- **Configurable Parameters**: Input paths, number of PCA components, and task selection are managed via `config2.yaml`.
- **Custom Visualizations**:
  - Explained variance bar charts.
  - False-colored PCA component images.
  - Loading vectors for selected principal components.

## Installation

### Requirements

- Python 3.8 or higher
- The following Python libraries:
  - `numpy`
  - `matplotlib`
  - `PyYAML`
  - `scikit-learn`
  - `opencv-python`

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sonainjameel/Industrial_Project_2024_Part_04.git
   cd Industrial_Project_2024_Part_04
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Input Data**: Ensure your `.hdr` and `.raw` files are placed in the folders specified in `config2.yaml`.

## Configuration

The project uses a `config2.yaml` file to define the tasks and their parameters. Below is an example configuration:

```yaml
tasks:
  - name: structure_oil
    dir_path: "path/to/oil/data"
    components: 10

  - name: structure_charring
    dir_path: "path/to/charring/data"
    components: 10

  - name: oil_charring
    dir_path: "path/to/oil_charring/data"
    components: 10

  - name: cracking
    dir_path: "path/to/cracking/data"
    components: 10
```

## Usage

1. **Run All Tasks**:
   To process all tasks defined in the `config2.yaml`:
   ```bash
   python3 pca_analysis.py
   ```

2. **Run a Specific Task**:
   To run only a specific task (e.g., `structure_oil`):
   ```bash
   python3 pca_analysis.py --task structure_oil
   ```

## Example Outputs

- **Structure Oil Analysis**:
  - PCA plots showing oil levels across different structures.
  - Loading vector visualizations with annotated spectral regions.

- **Charring Analysis**:
  - Visualizations of charred regions in PCA space.

## Project Structure

```
Industrial_Project_2024_Part_04/
├── config2.yaml               # Configuration file for tasks and parameters
├── pca_analysis.py    # Script for PCA processing and plotting
├── requirements.txt          # Required libraries
```

## Available Tasks

- **structure_oil**: Analyzes spectral data for oil content across structures.
- **structure_charring**: Examines spectral data for charring levels.
- **oil_charring**: Investigates the interaction of oil content and charring.
- **cracking**: Focuses on spectral data related to material cracking.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with a clear message.
4. Submit a pull request for review.

## License

This project is licensed under the MIT License.

## Acknowledgements

Special thanks to Sonain, Kasem, and Turab for their efforts, and to Joni Hyttinen and Prof. Markku Keinänen for their guidance and support throughout the project.
