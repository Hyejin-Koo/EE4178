프로젝트 시 참고하실 만한 코드를 안내합니다.

### 시간 확인

```python
import time

#Training..
start = time.time() # Train 시작 시간 정보 저장
...
end = time.time() # Train 종료 시간 정보 저장

duration = end - start # 종료 시간 - 시작 시간
print("Training takes {:.2f}minutes".format(duration/60)) #초 단위로 저장되므로, 60으로 나누어 분으로 표시
```


### GPU 확인
```python
!nvidia-smi
```
![image](https://user-images.githubusercontent.com/52940511/101285236-73a4eb00-3827-11eb-8d07-6e433d4fde29.png)
