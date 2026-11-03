export MUJOCO_GL=egl
export P="--pick_gen $HOME/cross_embody/fr3_pick_gen/scripts --scene_module build_scene --cam_name d435"
export A="--ee_site tcp_site --cube object --table_geom table_top --width 768 --height 432"
export CK=checkpoints/decision_core_vision_xyz_K10_bi_l10_h8_s42_x3s4.pt
export G="behaviour|yaw chain|grasped episodes|vision:|goal:|==>|UR5e:"
export E="python decision_core/eval_pp_vision.py $P $A --ckpt $CK --test_scene --episodes 50"
