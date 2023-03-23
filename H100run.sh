clear(){
    for i in {0..9};
    do echo "";
    done
}

set +x 

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_tinyanchor.py --amp
# clear



# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_GA.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_GA_DCNv2.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_GA_DCNv2_DH.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_EIOU.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_DH.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_DCNv2.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_dyhead.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_tinyanchor_GA_DCNv2_DH.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_CIOU.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_GIOU.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_EIOU.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_Guided.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_Guidedtiny.py --amp
# clear


# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_pre_tinyanchor.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_pre_EIOU.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_pre_tinyanchor_EIOU.py --amp
# clear


# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_pre_tinyanchor_SKIPAFPN.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_pre_tinyanchor_SKIPAFPN_EIOU.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_SKIPAFPN.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_pre_tinyanchor_SKIPAFPN_GA_DCNv2_DH_EIOU.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_pre_tinyanchor_SKIPAFPN_GA_DCNv2_DH.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_PAFPN_CARAFE_Skip_Parallel_concat-k1.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_PAFPN_CARAFE_Skip_Parallel_concat-k3.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_tinyanchor_PAFPN-cesk-cat-k1.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_GA12.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_GA1234.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_tinyanchor_PAFPN-cesk-cat-k1_GA_DCNv2.py --amp
# clear

# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_tinyanchor_PAFPN-cesk-cat-k1_GA_DCNv2_DH.py --amp
# clear


# python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_tinyanchor_SKIPAFPN_GA_DCNv2_DH.py --amp
# clear


python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_rsb/cascade-rcnn_r50_fpn_1x_rsb_tinyanchor_SCPAFPN.py --amp
clear

python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_rsb/cascade-rcnn_r50_fpn_1x_rsb_tinyanchor_SCPAFPN_EIOU.py --amp
clear

python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_rsb/cascade-rcnn_r50_fpn_1x_rsb_tinyanchor_SCPAFPN_GA.py --amp
clear

python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_rsb/cascade-rcnn_r50_fpn_1x_rsb_tinyanchor_SCPAFPN_DCNv2.py --amp
clear

python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_rsb/cascade-rcnn_r50_fpn_1x_rsb_tinyanchor_SCPAFPN_GA_DCNv2.py --amp
clear

python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_rsb/cascade-rcnn_r50_fpn_1x_rsb_tinyanchor_SCPAFPN_GA_DCNv2_DH.py --amp
clear

python tools/train.py myconfig/VisDrone-H100/cascade_rcnn_rsb/cascade-rcnn_r50_fpn_1x_rsb_tinyanchor_SCPAFPN_GA_DCNv2_DH_EIOU.py --amp
clear
