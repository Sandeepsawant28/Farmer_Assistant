metadata
license: cc-by-nc-nd-4.0
language:
  - kok
  - knn
  - gom
task_categories:
  - automatic-speech-recognition
pretty_name: Konkani ASR Dataset v0
dataset_info:
  features:
    - name: id
      dtype: string
    - name: audio
      dtype:
        audio:
          sampling_rate: 48000
    - name: text
      dtype: string
  splits:
    - name: train
      num_bytes: 5567183457
      num_examples: 201
  download_size: 5564222416
  dataset_size: 5567183457
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*
