# TFX MovieLens Recommender Pipeline

Production-style TFX pipeline for predicting user movie preferences using the MovieLens 100k dataset.

## Project Overview

Two-tower dot product model with embedding-based features for users and movies.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) Kubeflow Pipelines cluster

## Quick Start

1. Setup environment: Run setup_env.bat (Windows) or setup_env.sh (Linux/Mac)
2. Install dependencies: pip install -r requirements.txt
3. Launch notebook: jupyter notebook tfx_pipeline.ipynb

## Project Structure

```
TFX recommender/
├── ml-100k/                    # MovieLens 100k dataset
├── tfx_pipeline.ipynb         # Main pipeline notebook
├── transform_module.py        # TFX preprocessing
├── trainer_module.py          # Model definition
├── data/                      # Processed data
├── tfx_pipeline_output/      # Pipeline artifacts
├── requirements.txt          # Python dependencies
├── setup_env.bat            # Windows setup script
└── setup_env.sh             # Linux/Mac setup script
```
└── README.md                # This file
```

## Pipeline Components

1. **CsvExampleGen**: Ingest CSV data with time-based splits
2. **StatisticsGen**: Generate data statistics for analysis
3. **SchemaGen**: Infer feature schema
4. **ExampleValidator**: Validate data quality
5. **Transform**: Feature preprocessing and vocabulary creation
6. **Trainer**: Train two-tower model with embeddings
7. **Evaluator**: ROC-AUC + threshold-based metrics (0.3, 0.5, 0.7)
8. **Pusher**: Deploy model to serving directory

## Expected Performance

**Baseline Targets**:
- ROC-AUC: > 0.70
- F1-Score: > 0.65
- Precision/Recall: > 0.60

## 🎓 Usage

### Run Pipeline Locally

Open `tfx_pipeline.ipynb` and execute cells sequentially:

1. **Data Exploration** (Cells 1-4): Load and explore MovieLens 100k
2. **Data Preparation** (Cells 5-7): Time-based split and feature joining
3. **Define Components** (Cells 8-10): Create Transform and Trainer modules
4. **Run Pipeline** (Cell 11): Execute with LocalDagRunner
5. **Evaluate** (Cells 12-13): Analyze threshold-based metrics
6. **Deploy to Kubeflow** (Cells 14-16): Optional production deployment

### Run Pipeline Programmatically

```python
from tfx.orchestration.local.local_dag_runner import LocalDagRunner

# After defining pipeline in notebook
LocalDagRunner().run(tfx_pipeline)
```

### Evaluate Model

```python
results = evaluate_model_with_thresholds(
    model_path='./tfx_pipeline_output/serving_model',
    test_data_path='./data/test.csv'
)
```

## ☁️ Kubeflow Deployment

### Compile Pipeline

```python
kubeflow_yaml = create_kubeflow_pipeline()
```

### Deploy to Cluster

```bash
# Using kubectl
kubectl apply -f movielens-recommender-pipeline.yaml

# Or upload YAML to Kubeflow Pipelines UI
```

### Deploy with SDK

```python
import kfp

client = kfp.Client(host='http://your-kubeflow-endpoint:8080')
run = deploy_to_kubeflow_sdk(
    pipeline_yaml='movielens-recommender-pipeline.yaml',
    kubeflow_endpoint='http://your-kubeflow-endpoint:8080'
)
```

## 🔍 Key Implementation Details

### Time-Based Split
- **Train**: 70% earliest interactions (historical data)
- **Validation**: 15% middle interactions
- **Test**: 15% most recent interactions
- **Rationale**: Simulates production (predict future from past)

### Embedding Dimensions
- User ID: 32
- Movie ID: 32
- Age (6 buckets): 12
- Gender (M/F): 6
- Occupation (21 categories): 12
- Genre (19 genres, averaged): 12
- Final vectors: 64 (both towers)

### Evaluation Metrics
- **ROC-AUC**: Threshold-independent ranking quality
- **Precision**: What % of predicted "likes" are correct
- **Recall**: What % of actual likes are predicted
- **F1-Score**: Harmonic mean of precision and recall
- **Thresholds**: 0.3 (high recall), 0.5 (balanced), 0.7 (high precision)

## 🛠️ Troubleshooting

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### CUDA/GPU Issues
```bash
# Use CPU-only TensorFlow
pip install tensorflow-cpu>=2.12.0
```

### TFX Component Errors
- Check module files exist: `transform_module.py`, `trainer_module.py`
- Verify data files: `./data/train.csv`, `./data/val.csv`, `./data/test.csv`
- Clear cache: Delete `tfx_pipeline_output/` and re-run

## 📚 References

- [TFX Documentation](https://www.tensorflow.org/tfx)
- [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/100k/)
- [Two-Tower Models](https://developers.google.com/machine-learning/recommendation/dnn/retrieval)

## 📝 Citation

```
F. Maxwell Harper and Joseph A. Konstan. 2015. 
The MovieLens Datasets: History and Context. 
ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19.
DOI=http://dx.doi.org/10.1145/2827872
```

## 📄 License

This project uses the MovieLens 100k dataset. Please cite the dataset when publishing results.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📧 Contact

For questions or issues, please open an issue in the repository.
