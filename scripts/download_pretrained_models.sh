mkdir -p models/ek55_0.5
cd models/ek55_0.5

curl https://transfer.sh/brX/RULSTM-anticipation_0.5_6_4_fusion_best.pth.tar -o RULSTM-anticipation_0.5_6_4_fusion_best.pth.tar
curl https://transfer.sh/1b46vvL/RULSTM-anticipation_0.5_6_4_flow_best.pth.tar -o RULSTM-anticipation_0.5_6_4_flow_best.pth.tar
curl https://transfer.sh/1VflKgK/RULSTM-anticipation_0.5_6_4_obj_best.pth.tar -o RULSTM-anticipation_0.5_6_4_obj_best.pth.tar
curl https://transfer.sh/1VvVhxj/RULSTM-anticipation_0.5_6_4_rgb_best.pth.tar -o RULSTM-anticipation_0.5_6_4_rgb_best.pth.tar

cd ../../
mkdir -p models/ek55_0.125
cd models/ek55_0.125

curl https://transfer.sh/1tYUQD7/RULSTM-anticipation_0.125_24_16_flow_best.pth.tar -o RULSTM-anticipation_0.125_24_16_flow_best.pth.tar
curl https://transfer.sh/12Q1oqp/RULSTM-anticipation_0.125_24_16_obj_best.pth.tar -o RULSTM-anticipation_0.125_24_16_obj_best.pth.tar
curl https://transfer.sh/1uEWmXy/RULSTM-anticipation_0.125_24_16_rgb_best.pth.tar -o RULSTM-anticipation_0.125_24_16_rgb_best.pth.tar
----------------curl https://transfer.sh/1VvVhxj/RULSTM-anticipation_0.5_6_4_rgb_best.pth.tar -o RULSTM-anticipation_0.5_6_4_rgb_best.pth.tar

cd ../../
mkdir -p models/ek55
cd models/ek55


