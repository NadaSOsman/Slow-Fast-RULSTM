# SlowFast Rolling-Unrolling LSTMs for Action Anticipation in Egocentric Videos
This repository hosts the code related to the paper:
Osman, Nada, Guglielmo Camporese, Pasquale Coscia, and Lamberto Ballan. "SlowFast Rolling-Unrolling LSTMs for Action Anticipation in Egocentric Videos." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3437-3445. 2021 [Download](https://openaccess.thecvf.com/content/ICCV2021W/EPIC/papers/Osman_SlowFast_Rolling-Unrolling_LSTMs_for_Action_Anticipation_in_Egocentric_Videos_ICCVW_2021_paper.pdf)

```
@inproceedings{osman2021slowfast,
  title={SlowFast Rolling-Unrolling LSTMs for Action Anticipation in Egocentric Videos},
  author={Osman, Nada and Camporese, Guglielmo and Coscia, Pasquale and Ballan, Lamberto},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3437--3445},
  year={2021}
}
```
# Datasets and Environment Preperations
Follow the instructions privided in [RULSTM](https://github.com/fpv-iplab/rulstm) to download the EPIC-KITCHENs-55 and EGTEA-Gaze datasets and to setup the envirovment.

# Model
### Model architectures
We define too model architures for the slow-fast fusion:

#### Architecture #1:
```python
                                   ↑
                           ModalitiesFusionArc1
                                   ↑
            ┌ ------------------------------------------ ┐
            ↑                      ↑                     ↑
    SlowFastFusionArc1     SlowFastFusionArc1     SlowFastFusionArc1
            ↑                      ↑                     ↑
       ┌ ------- ┐            ┌ -------- ┐           ┌ ------- ┐
       ↑         ↑            ↑         ↑            ↑         ↑   
    RGB-Slow  RGB-Fast     Obj-Slow  Obj-Fast     Flow-Slow  Flow-Fast
```

Stages

* For each modality:
  * train the `Slow` branch with `sequence completion`,
  * finetune the `Slow` branch,
  * train the `Fast` branch with `sequence completion`,
  * finetune the `Fast` branch,
  * train the `SlowFastFusionArc1` with the `Slow` and `Fast` branches,
* train the `ModalitiesFusionArc1` passing all the `SlowFastFusionArc1` modalities.

#### Architecture #2:
```python
                                   ↑
                           SlowFastFusionArch2
                                   ↑
                 ┌ -------------------------------- ┐
                 ↑                                  ↑
       ModalitiesFusionArc2               ModalitiesFusionArc2
                 ↑                                  ↑
       ┌ ----------------- ┐              ┌ ----------------- ┐
       ↑         ↑         ↑              ↑         ↑         ↑
    RGB-Slow  Obj-Slow  Flow-Slow      RGB-Fast  Obj-Fast  Flow-Fast
```

Stages

* For each frame rate:
  * train the `RGB-Slow` branch with `sequence completion`
  * train the `Obj-Slow` branch with `sequence completion`,
  * train the `Flow-Slow` branch with `sequence completion`,
  * finetune the `RGB-Slow`,
  * finetune the `Obj-Slow`,
  * finetune the `Flow-Slow`,
  * finetune the `Slow` branch,
* train the `ModalitiesFusionArc2` with for the `Slow` and `Fast` branches,
* train the `SlowFastFusionArch2` with the `Slow` and `Fast` branches.

# Training
1. Single-modality Single-timescale training:
```
python main.py train data/ek55 models/ek55 --modality rgb --task anticipation --sequence_completion
```
```
python main.py train data/ek55 models/ek55 --modality rgb --task anticipation
```
2. Repeat for all modalities (rgb/flow/obj), and all timescales. For the obj modality, set ```--feat_in 352```
3. Slow-Fast Fusion Model Arch1:
  ..a. Run Slow-Fast Fusion on a single modality:
  ```
  python main.py train data_path models/ek55 --modality rgb --task anticipation --slowfastfusion --alphas_fused 0.125 0.5 --S_enc_fused 24 6 --S_ant_fused 16 4
  ```
  ..b. Repeat for all modalities.
  ..c. Run Modalities fusion with Arch1:
  ```
  python main.py train data_path models/ek55 --modality fusion --task anticipation --slowfastfusion --arc1 --alphas_fused 0.125 0.5 --S_enc_fused 24 6 --S_ant_fused 16 4 --dropout 0.9
  ```
4. Slow-Fast Fusion Model Arch2:
  ..a. Run Modalities fusion with slow timescale
  ```
  python main.py train data_path models/ek55 --modality fusion --task anticipation --alpha 0.5 --S_enc 6 --S_ant 4
  ```
  ..b. Run Modalities fusion with fast timescale
  ```
  python main.py train data_path models/ek55 --modality fusion --task anticipation --alpha 0.125 --S_enc 24 --S_ant 16
  ```
  ..c. Run slow-fast fusion on the fused modalites
  ```
  python main.py train data_path models/ek55 --modality fusion --task anticipation --slowfastfusion --alphas_fused 0.125 0.5 --S_enc_fused 24 6 --S_ant_fused 16 4 --dropout 0.9
  ```
5. To validate a trained model, run ```python main.py validate``` with the same options as in the training command
6. To test a trained model, run ```python main.py test --json_directory jsons/ek55``` with the same options as in the training command.
