# Kaggle Vasculature 3D segmentation Competition

## Get Started

### Environment

vessl 사용할 경우 pytorch 22.12 image로 workspace deploy 하면 됩니다.

```
python 3.8+
```

### Download dataset

vessl에서 Vasculature3D dataset을 다운로드해야합니다.
root 디렉토리는 용량 제한이 있으므로 top directory에 다운로드 해야합니다.

```sh
vessl dataset download Vasculature3D / /
```

### How to run

run.sh 에 실험할 하이퍼파라미터를 설정해 실행 스크립트를 작성한 후 실행할 수 있습니다.

```
# authorize shell script
chmod +x run.sh
# run
./run.sh
```
