# Slow-Fast-RULSTM

1- Get Pretrained models:
  Use the the script ```'download_pretrained_models.sh'``` to get the pretrained models for 0.5 and 0.125 alpha values and the fusion models.

2- Train the final fusion modesl (currently the final model fuses the slow modality fused models [rgb-flow-obj @ 0.5] and the fast modality fused models [@ 0.125]):

```
python3 main.py train data_path models/ek55 --modality fusion --task anticipation --slowfastfusion --alphas_fused 0.125 0.5 --S_enc_fused 24 6 --S_ant_fused 16 4 --dropout 0.9
```

This model has some trained epochs saved (30 epochs with best acc 35.46). to resume after that, run the same command with the addition of ```'--resume'```

3- To validate the final model, run the command:

```
python3 main.py validate data_path models/ek55 --modality fusion --task anticipation --slowfastfusion --alphas_fused 0.125 0.5 --S_enc_fused 24 6 --S_ant_fused 16 4 --dropout 0.9
```

# Model
### Model architectures

Let's define the two kind of architectures:

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
