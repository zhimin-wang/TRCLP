# Cross User

for windowSize in 2
do
    for fold in 01 02 03 04 05
    do
        let item=$windowSize
        let number=$item*25*2
        let allnum=$number*3
        baseJobName="--checkpoint ../checkpoint/20_Person_Cross_User_5_Fold_"$item"_WinSize_OursMethod/Fold_"$fold"/ --prdDir ../predictions/20_Person_Cross_User_5_Fold_"$item"_WinSize_OursMethod/Test_Fold_"$fold"/ --datasetDir ../../Person_20/Cross_User_5_Fold_"$item"_WinSize_all_results/Fold_"$fold"/"
        perCommand="python3 TRCLP.py --augmode 2 --demo 0 --test 0 --encoderEpochs 30 --pretrainFlag 1 --trainClassifierFlag 0  --classiferEpochs 30 --savePrd 0 --inputSize "$allnum" --eyeFeatureSize "$number" --headFeatureSize "$number" --gwFeatureSize "$number" --interval 1 --batchSize 256 --learningRate 0.01 --gamma 0.75 --weightDecay 1e-4 --numClasses 4 --temp 0.07 "$baseJobName
        echo $perCommand
        CUDA_VISIBLE_DEVICES="0" command $perCommand

        let item=$windowSize
        let number=$item*25*2
        let allnum=$number*3
        baseJobName="--checkpoint ../checkpoint/20_Person_Cross_User_5_Fold_"$item"_WinSize_OursMethod/Fold_"$fold"/ --prdDir ../predictions/20_Person_Cross_User_5_Fold_"$item"_WinSize_OursMethod/Test_Fold_"$fold"/ --datasetDir ../../Person_20/Cross_User_5_Fold_"$item"_WinSize_all_results/Fold_"$fold"/"
        perCommand="python3 TRCLP.py --augmode 2 --demo 0 --test 1 --encoderEpochs 30 --pretrainFlag 0 --trainClassifierFlag 1  --classiferEpochs 30 --savePrd 0 --inputSize "$allnum" --eyeFeatureSize "$number" --headFeatureSize "$number" --gwFeatureSize "$number" --interval 1 --batchSize 256 --learningRate 0.01 --gamma 0.75 --weightDecay 1e-4 --numClasses 4 --temp 0.07 "$baseJobName
        echo $perCommand
        CUDA_VISIBLE_DEVICES="0" command $perCommand
    done
done
