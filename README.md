# Smartphone Colorimetry and Machine Learning for Borax Screening

Course project for **Project Mid Term — Prof. Dr. Guruh Fajar Shidik**.

This repository contains the Python analysis program and its required Python packages. The final paper is submitted separately through the course assignment link and is not presented here as a published article.

## Complete assignment materials

The complete set of course materials is available in the Google Drive folder below. It includes the IEEE-formatted paper (Word and PDF), Turnitin result, explanation video, raw data, program output archive, and presentation slides.

[Open the complete assignment folder on Google Drive](https://drive.google.com/drive/folders/1ncXH3nr6dHyIJHY806kW4HJqN6ORCK-F?usp=sharing)

The explanation video is also available on YouTube: https://youtu.be/0zlsuIgrtno

## Repository files

- `Program_ML_Colorimetry_Mada_Sanjaya_WS-P36.2026.00011.py` — analysis and figure/table generation script.
- `requirements.txt` — Python packages needed by the script.

## Data folders expected by the program

Place the raw data beside the Python script, using the folder names expected by the program:

- `ganjil/` — odd-concentration smartphone images.
- `genap/` — even-concentration smartphone images.
- `Data Spectro Ganjil/` — optional odd-set UV–Vis files.
- `Data Spectro Genap/` — optional even-set UV–Vis files.

The raw data are provided in the Google Drive folder above and are not committed here because the source archive is large.

## Run

Install the dependencies and run the script from the project folder:

```bash
python -m pip install -r requirements.txt
python Program_ML_Colorimetry_Mada_Sanjaya_WS-P36.2026.00011.py
```

The script writes its generated figures, tables, processed data, and model files to the output folder specified inside the program.

## Video

The YouTube video serves as both the experiment demonstration and the explanation video. Confirm that the Drive folder and video are accessible to the instructor before submitting this repository URL.
