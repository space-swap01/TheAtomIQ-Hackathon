@echo off
echo ===================================================
echo Starting QuantumBank Streamlit Dashboard...
echo Open your browser at http://localhost:8501
echo ===================================================
"C:\ProgramData\Anaconda3\python.exe" -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
pause
