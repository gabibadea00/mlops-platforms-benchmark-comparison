# Run scripts

## MLflow - Adult Income dataset

```bash
    python3 MLflow-Adult.py

    mlflow ui --host 127.0.0.1 --port 5001
```
## MLflow - Deep Covid dataset

```bash
    python3 ResNet18_train.py --dataset_path ./../../../../datasets/DeepCovid/data_upload_v3 --batch_size 20 --epoch 10 --num_workers 8 --learning_rate 0.001

    mlflow ui --host 127.0.0.1 --port 5002
```

## MetaFlow - Adult Income dataset

```bash
    python3 MetaFlow-Adult.py run --with card
    # http://localhost:8324 
    python3 MetaFlow-Adult.py card server
    # python3 MetaFlow-Adult.py card view start
```

## MetaFlow - Deep Covid dataset

```bash
    python3 ResNet18_train.py --environment=conda run --with card
    # python3 ResNet18_train.py run --with card
    # http://localhost:8324 
    python3 ResNet18_train.py --environment=conda card server
    # python3 ResNet18_train.py --environment=conda card view start
```

## ZenML - Adult Income dataset

```bash
    zenml init
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    zenml login --local
    python3 ZenML-Adult.py
    zenml logout --local

        zenml server up –> pornește serverul (daemon sau Docker).
        zenml server connect <nume_server> –> conectează clientul local la server.
        zenml server logs <nume_server> –> vizualizează log-urile serverului.
        zenml server down –> oprește serverul.
```

## ZenML - Deep Covid dataset

```bash
    zenml init
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    zenml login --local
    python3 ResNet18_train.py --dataset_path ./../../../../datasets/DeepCovid/data_upload_v3 --batch_size 20 --epoch 10 --num_workers 8 --learning_rate 0.001
    zenml logout --local

        zenml server up –> pornește serverul (daemon sau Docker).
        zenml server connect <nume_server> –> conectează clientul local la server.
        zenml server logs <nume_server> –> vizualizează log-urile serverului.
        zenml server down –> oprește serverul.
```