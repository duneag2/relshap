# RelShap: Relationally Consistent Shapley Explanations

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
bash run_all.sh
```

Most experiment settings are configured in `run_all.sh`, including the dataset, model, random seeds, and RelShap options.

## Main Files

- `run_all.sh`: runs the full experiment pipeline.
- `run_model.py`: trains and saves ML/DL models.
- `run_relshap.py`: computes RelShap explanations.
- `constraint_schema.py`: extracts constraints from the database schema.
- `constraint_query.py`: extracts constraints from SQL queries.
- `constraint_data.py`: extracts constraints from data.
- `fd_ic_refinement.py`: combines and refines constraints.
- `dataset/`: dataset-specific scripts, SQL files, configs, logs, and outputs.

## Notes

- Logs are written under each dataset's `logs/` directory.
