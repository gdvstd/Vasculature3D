# Kaggle Vasculature 3D segmentation Competition

## Get Started

### Environment

vessl 사용할 경우 pytorch 22.12 image로 workspace deploy 하면 됩니다.

- python 3.8+

이제 프로젝트 폴더에 repository를 git clone합니다. **원본 repository를 clone하면 push시 충돌이 발생할 수 있으므로 Fork 후 clone할 것을 강력히 권장합니다.**
```
git clone [URL]
cd Vasculature3D
```

### Download dataset

vessl에서 Vasculature3D dataset을 다운로드합니다.

root 디렉토리는 용량 제한이 있으므로 top directory에 다운로드 해야합니다. 아래 명령어는 백그라운드에서 다운로드 프로세스가 돌게 해줍니다.

```sh
nohup vessl dataset download Vasculature3D / / > download.log 2>&1 &
```

### How to run

아래 명령어를 통해 현재 GPU에 가용한 용량을 확인할 수 있습니다. 

```sh
nvidia-smi
```

run.sh 에 실험할 조건을 작성해 실행할 수 있습니다. CUDA_VISIBLE_DEVICES 환경변수로 프로세스를 올릴 GPU를 지정합니다.

여러 GPU를 사용하면서 동시에 몇가지 실험을 돌릴 수 있습니다.

```sh
# run on gpu index 0
CUDA_VISIBLE_DEVICES=0 sh run.sh
```

### how to kill process 

kill_python script를 이용해 run 중인 python process를 확인하고 지정하여 정지할 수 있습니다.

```bash
bash kill_python.bash
```

#### Manual Method

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

위 방법으로 running process를 확인한 후 python으로 실행된 process의 Process ID(PID)를 지정해 kill 해줍니다.
강제종료하지 않으면 해제되지 않은 메모리가 남을 수 있으므로 강제종료(-9)해야합니다.

```
kill -9 [PID]
```



