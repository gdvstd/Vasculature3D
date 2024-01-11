# Kaggle Vasculature 3D segmentation Competition

## Get Started

### Environment

vessl 사용할 경우 pytorch 22.12 image로 workspace deploy 하면 됩니다.

- Mac OS Sonoma 14.2
- python 3.8+

이제 내 프로젝트 폴더에 repository를 git clone합니다. 원본 repository를 clone하면 push시 충돌이 발생할 수 있으므로 Fork 후 clone할 것을 권장합니다.
```
git clone
cd Vasculature3D
```

### Download dataset

vessl에서 Vasculature3D dataset을 다운로드합니다.

root 디렉토리는 용량 제한이 있으므로 top directory에 다운로드 해야합니다.

```sh
vessl dataset download Vasculature3D / /
```

### How to run

run.sh 에 실험할 조건을 작성해 실행할 수 있습니다.

CUDA_VISIBLE_DEVICES 환경변수로 프로세스를 올릴 GPU를 지정합니다.

여러 DEVICE를 사용하면 동시에 여러 실험을 돌릴 수 있습니다.

```sh
# authorize shell script
chmod +x run.sh
# run on gpu index 0
CUDA_VISIBLE_DEVICES=0 ./run.sh
```

### how to kill process 

run.sh 스크립트는 백그라운드에서 돌아가도록 설정되어 있으므로 (nohup) 실험을 멈추려면 프로세스를 직접 정지해줘야 합니다.

ps는 linux system의 running process를 display 해주는 명령어입니다.
아래는 full format으로 running process의 정보를 보여줍니다.
```sh
ps -ef
```

아래처럼 grep을 이용해 python이 포함된 line만 필터링 할 수도 있습니다.
```
ps -ef | grep 'python'
```

top은 running process에 대한 정보를 동적으로 보여줍니다.
```sh
top
```

위 방법으로 running process에 대한 정보를 얻을 수 있습니다. 



