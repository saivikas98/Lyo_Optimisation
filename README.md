# Primary Drying Streamlit App

## Files
- `app.py`: Streamlit application
- `requirements.txt`: Python dependencies

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app will open in a browser, usually at `http://localhost:8501`.

## Notes
- The original hard-coded Windows path and direct workbook writing were removed.
- Simulation results can be downloaded from the app as CSV or Excel.
- The model includes a maximum duration, rejected-step limit, and solver-failure limit to avoid an endless loop.
- Validate all units and model assumptions before using results for process decisions.
