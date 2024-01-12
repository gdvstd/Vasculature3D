# 잘 바꾸지 않는 config
SAVE_DIR='./save'
LOG_DIR='./logs'

if [ ! -d "$SAVE_DIR" ]; then
    mkdir $SAVE_DIR
fi
if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

# 바뀌는 config
BATCH=16
PATCH=64
SAVE_FREQUENCY=200

EXPERIMENT="test"
LOG_FILE="./logs/$EXPERIMENT.txt"

echo "Batch: $BATCH" > $LOG_FILE
echo "Patch: $PATCH" >> $LOG_FILE
echo "Save Frequency: $SAVE_FREQUENCY" >> $LOG_FILE

nohup python main.py -n $EXPERIMENT -b $BATCH -p $PATCH -sf $SAVE_FREQUENCY >> $LOG_FILE 2>&1 &