for windowSize in 2
do
    for fold in 01
    do
        let item=$windowSize
        let number=$item*25*2
        let allnum=$number*3
        baseJobName="--checkpoint ../checkpoint/pretrained_cross_user/Fold_"$fold"/ --prdDir ../predictions/cross_user/Test_Fold_"$fold"/ --datasetDir ../../Person_20/Cross_User_5_Fold_"$item"_WinSize_all_results/Fold_"$fold"/"
        perCommand="python3 TRCLP.py --runEncoder 30 --runClassifer 28 --augmode 0 --demo 0 --test 1 --encoderEpochs 30 --pretrainFlag 0 --trainClassifierFlag 0  --classiferEpochs 30 --savePrd 1 --onlyFilter 1 --inputSize "$allnum" --eyeFeatureSize "$number" --headFeatureSize "$number" --gwFeatureSize "$number" --interval 1 --batchSize 256 --learningRate 0.01 --gamma 0.75 --weightDecay 1e-4 --numClasses 4 --temp 0.07 "$baseJobName
        echo $perCommand
        CUDA_VISIBLE_DEVICES="0" command $perCommand
    done

    for fold in 02
    do
        let item=$windowSize
        let number=$item*25*2
        let allnum=$number*3
        baseJobName="--checkpoint ../checkpoint/pretrained_cross_user/Fold_"$fold"/ --prdDir ../predictions/cross_user/Test_Fold_"$fold"/ --datasetDir ../../Person_20/Cross_User_5_Fold_"$item"_WinSize_all_results/Fold_"$fold"/"
        perCommand="python3 TRCLP.py --runEncoder 30 --runClassifer 12 --augmode 0 --demo 0 --test 1 --encoderEpochs 30 --pretrainFlag 0 --trainClassifierFlag 0  --classiferEpochs 30 --savePrd 1 --onlyFilter 1 --inputSize "$allnum" --eyeFeatureSize "$number" --headFeatureSize "$number" --gwFeatureSize "$number" --interval 1 --batchSize 256 --learningRate 0.01 --gamma 0.75 --weightDecay 1e-4 --numClasses 4 --temp 0.07 "$baseJobName
        echo $perCommand
        CUDA_VISIBLE_DEVICES="0" command $perCommand
    done

    for fold in 03
    do
        let item=$windowSize
        let number=$item*25*2
        let allnum=$number*3
        baseJobName="--checkpoint ../checkpoint/pretrained_cross_user/Fold_"$fold"/ --prdDir ../predictions/cross_user/Test_Fold_"$fold"/ --datasetDir ../../Person_20/Cross_User_5_Fold_"$item"_WinSize_all_results/Fold_"$fold"/"
        perCommand="python3 TRCLP.py --runEncoder 15 --runClassifer 13 --augmode 0 --demo 0 --test 1 --encoderEpochs 30 --pretrainFlag 0 --trainClassifierFlag 0  --classiferEpochs 30 --savePrd 1 --onlyFilter 1 --inputSize "$allnum" --eyeFeatureSize "$number" --headFeatureSize "$number" --gwFeatureSize "$number" --interval 1 --batchSize 256 --learningRate 0.01 --gamma 0.75 --weightDecay 1e-4 --numClasses 4 --temp 0.07 "$baseJobName
        echo $perCommand
        CUDA_VISIBLE_DEVICES="0" command $perCommand
    done

    for fold in 04
    do
        let item=$windowSize
        let number=$item*25*2
        let allnum=$number*3
        baseJobName="--checkpoint ../checkpoint/pretrained_cross_user/Fold_"$fold"/ --prdDir ../predictions/cross_user/Test_Fold_"$fold"/ --datasetDir ../../Person_20/Cross_User_5_Fold_"$item"_WinSize_all_results/Fold_"$fold"/"
        perCommand="python3 TRCLP.py --runEncoder 15 --runClassifer 12 --augmode 0 --demo 0 --test 1 --encoderEpochs 30 --pretrainFlag 0 --trainClassifierFlag 0  --classiferEpochs 30 --savePrd 1 --onlyFilter 1 --inputSize "$allnum" --eyeFeatureSize "$number" --headFeatureSize "$number" --gwFeatureSize "$number" --interval 1 --batchSize 256 --learningRate 0.01 --gamma 0.75 --weightDecay 1e-4 --numClasses 4 --temp 0.07 "$baseJobName
        echo $perCommand
        CUDA_VISIBLE_DEVICES="0" command $perCommand
    done

    for fold in 05
    do
        let item=$windowSize
        let number=$item*25*2
        let allnum=$number*3
        baseJobName="--checkpoint ../checkpoint/pretrained_cross_user/Fold_"$fold"/ --prdDir ../predictions/cross_user/Test_Fold_"$fold"/ --datasetDir ../../Person_20/Cross_User_5_Fold_"$item"_WinSize_all_results/Fold_"$fold"/"
        perCommand="python3 TRCLP.py --runEncoder 30 --runClassifer 17 --augmode 0 --demo 0 --test 1 --encoderEpochs 30 --pretrainFlag 0 --trainClassifierFlag 0  --classiferEpochs 30 --savePrd 1 --onlyFilter 1 --inputSize "$allnum" --eyeFeatureSize "$number" --headFeatureSize "$number" --gwFeatureSize "$number" --interval 1 --batchSize 256 --learningRate 0.01 --gamma 0.75 --weightDecay 1e-4 --numClasses 4 --temp 0.07 "$baseJobName
        echo $perCommand
        CUDA_VISIBLE_DEVICES="0" command $perCommand
    done

done