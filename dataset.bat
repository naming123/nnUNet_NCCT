@echo off
REM 환경 변수 설정
set nnUNet_preprocessed=C:\Users\user\nnunet\nnUNet_CECT_padding0\nnUNet_preprocessed
set nnUNet_raw=C:\Users\user\nnunet\nnUNet_CECT_padding0\nnUNet_raw
set nnUNet_results=C:\Users\user\nnunet\nnUNet_CECT_padding0\nnUNet_results

REM 데이터셋 계획 및 전처리 실행
set CUDA_VISIBLE_DEVICES=0
nnUNetv2_plan_and_preprocess -d 500 --verify_dataset_integrity

REM 5개의 fold에 대해 반복 실행
for /L %%f in (0,1,4) do (
    echo Executing: CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 500 3d_fullres %%f
    set CUDA_VISIBLE_DEVICES=2
    nnUNetv2_train 500 3d_fullres %%f
)

pause
