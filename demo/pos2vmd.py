#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# pos2vmd.py - convert joint position data to VMD


import numpy as np
from PyQt5.QtGui import QQuaternion, QVector3D
from VmdWriter import VmdBoneFrame,VmdWriter

def positions_to_frames(index, pos, head_rotation=None):
    """convert positions to bone frames"""
    frames = []
    # 上半身
    bf = VmdBoneFrame(index)
    bf.name = b'\x8f\xe3\x94\xbc\x90\x67' # '上半身'
    direction = pos[8] - pos[7]
    up = QVector3D.crossProduct(direction, (pos[14] - pos[11])).normalized()
    upper_body_orientation = QQuaternion.fromDirection(direction, up)
    initial = QQuaternion.fromDirection(QVector3D(0, 1, 0), QVector3D(0, 0, 1))
    bf.rotation = upper_body_orientation * initial.inverted()
    frames.append(bf)
    upper_body_rotation = bf.rotation
    
    # 下半身
    bf = VmdBoneFrame(index)
    bf.name = b'\x89\xba\x94\xbc\x90\x67' # '下半身'
    direction = pos[0] - pos[7]
    up = QVector3D.crossProduct(direction, (pos[4] - pos[1]))
    lower_body_orientation = QQuaternion.fromDirection(direction, up)
    initial = QQuaternion.fromDirection(QVector3D(0, -1, 0), QVector3D(0, 0, 1))
    bf.rotation = lower_body_orientation * initial.inverted()
    lower_body_rotation = bf.rotation
    frames.append(bf)

    # 首は回転させず、頭のみ回転させる
    # 頭
    '''bf = VmdBoneFrame(index)
    bf.name = b'\x93\xaa'
    if head_rotation is None:
        # direction = pos[10] - pos[9]
        direction = pos[10] - pos[8]
        up = QVector3D.crossProduct((pos[9] - pos[8]), (pos[10] - pos[9]))
        orientation = QQuaternion.fromDirection(direction, up)
        initial_orientation = QQuaternion.fromDirection(QVector3D(0, 1, 0), QVector3D(1, 0, 0))
        rotation = orientation * initial_orientation.inverted()
        bf.rotation = upper_body_rotation.inverted() * rotation
    else:
        bf.rotation = upper_body_rotation.inverted() * head_rotation
    frames.append(bf)'''
        
    # 左腕
    bf = VmdBoneFrame(index)
    bf.name = b'\x8d\xb6\x98\x72' # '左腕'
    direction = pos[12] - pos[11]
    up = QVector3D.crossProduct((pos[12] - pos[11]), (pos[13] - pos[12]))
    orientation = QQuaternion.fromDirection(direction, up)
    initial_orientation = QQuaternion.fromDirection(QVector3D(1, -1, 0), QVector3D(1, 1, 0))
    rotation = orientation * initial_orientation.inverted()
    # 左腕ポーンの回転から親ボーンの回転を差し引いてbf.rotationに格納する。
    # upper_body_rotation * bf.rotation = rotation なので、
    bf.rotation = upper_body_rotation.inverted() * rotation
    left_arm_rotation = bf.rotation # 後で使うので保存しておく
    frames.append(bf)
    
    # 左ひじ
    bf = VmdBoneFrame(index)
    bf.name = b'\x8d\xb6\x82\xd0\x82\xb6' # '左ひじ'
    direction = pos[13] - pos[12]
    up = QVector3D.crossProduct((pos[12] - pos[11]), (pos[13] - pos[12]))
    orientation = QQuaternion.fromDirection(direction, up)
    initial_orientation = QQuaternion.fromDirection(QVector3D(1, -1, 0), QVector3D(1, 1, 0))
    rotation = orientation * initial_orientation.inverted()
    # 左ひじポーンの回転から親ボーンの回転を差し引いてbf.rotationに格納する。
    # upper_body_rotation * left_arm_rotation * bf.rotation = rotation なので、
    bf.rotation = left_arm_rotation.inverted() * upper_body_rotation.inverted() * rotation
    # bf.rotation = (upper_body_rotation * left_arm_rotation).inverted() * rotation # 別の表現
    frames.append(bf)

    
    # 右腕
    bf = VmdBoneFrame(index)
    bf.name = b'\x89\x45\x98\x72' # '右腕'
    direction = pos[15] - pos[14]
    up = QVector3D.crossProduct((pos[15] - pos[14]), (pos[16] - pos[15]))
    orientation = QQuaternion.fromDirection(direction, up)
    initial_orientation = QQuaternion.fromDirection(QVector3D(-1, -1, 0), QVector3D(1, -1, 0))
    rotation = orientation * initial_orientation.inverted()
    bf.rotation = upper_body_rotation.inverted() * rotation
    right_arm_rotation = bf.rotation
    frames.append(bf)
    
    # 右ひじ
    bf = VmdBoneFrame(index)
    bf.name = b'\x89\x45\x82\xd0\x82\xb6' # '右ひじ'
    direction = pos[16] - pos[15]
    up = QVector3D.crossProduct((pos[15] - pos[14]), (pos[16] - pos[15]))
    orientation = QQuaternion.fromDirection(direction, up)
    initial_orientation = QQuaternion.fromDirection(QVector3D(-1, -1, 0), QVector3D(1, -1, 0))
    rotation = orientation * initial_orientation.inverted()
    bf.rotation = right_arm_rotation.inverted() * upper_body_rotation.inverted() * rotation
    frames.append(bf)

    # 左足
    bf = VmdBoneFrame(index)
    bf.name = b'\x8d\xb6\x91\xab' # '左足'
    direction = pos[5] - pos[4]
    up = QVector3D.crossProduct((pos[5] - pos[4]), (pos[6] - pos[5]))
    orientation = QQuaternion.fromDirection(direction, up)
    initial_orientation = QQuaternion.fromDirection(QVector3D(0, -1, 0), QVector3D(-1, 0, 0))
    rotation = orientation * initial_orientation.inverted()
    bf.rotation = lower_body_rotation.inverted() * rotation
    left_leg_rotation = bf.rotation
    frames.append(bf)
    
    # 左ひざ
    bf = VmdBoneFrame(index)
    bf.name = b'\x8d\xb6\x82\xd0\x82\xb4' # '左ひざ'
    direction = pos[6] - pos[5]
    up = QVector3D.crossProduct((pos[5] - pos[4]), (pos[6] - pos[5]))
    orientation = QQuaternion.fromDirection(direction, up)
    initial_orientation = QQuaternion.fromDirection(QVector3D(0, -1, 0), QVector3D(-1, 0, 0))
    rotation = orientation * initial_orientation.inverted()
    bf.rotation = left_leg_rotation.inverted() * lower_body_rotation.inverted() * rotation
    frames.append(bf)

    # 右足
    bf = VmdBoneFrame(index)
    bf.name = b'\x89\x45\x91\xab' # '右足'
    direction = pos[2] - pos[1]
    up = QVector3D.crossProduct((pos[2] - pos[1]), (pos[3] - pos[2]))
    orientation = QQuaternion.fromDirection(direction, up)
    initial_orientation = QQuaternion.fromDirection(QVector3D(0, -1, 0), QVector3D(-1, 0, 0))
    rotation = orientation * initial_orientation.inverted()
    bf.rotation = lower_body_rotation.inverted() * rotation
    right_leg_rotation = bf.rotation
    frames.append(bf)
    
    # 右ひざ
    bf = VmdBoneFrame(index)
    bf.name = b'\x89\x45\x82\xd0\x82\xb4' # '右ひざ'
    direction = pos[3] - pos[2]
    up = QVector3D.crossProduct((pos[2] - pos[1]), (pos[3] - pos[2]))
    orientation = QQuaternion.fromDirection(direction, up)
    initial_orientation = QQuaternion.fromDirection(QVector3D(0, -1, 0), QVector3D(-1, 0, 0))
    rotation = orientation * initial_orientation.inverted()
    bf.rotation = right_leg_rotation.inverted() * lower_body_rotation.inverted() * rotation
    frames.append(bf)

    return frames



def convert_position(pose_3d):
    data=np.load(pose_3d)['reconstruction']
    positions = []
    for frame in data:
        p=[]
        for pos in frame:
            q = QVector3D(pos[0], -1.0*pos[1], pos[2])
            p.append(q)
        positions.append(p)
    return positions
    
def position_list_to_vmd(positions, vmd_file, head_rotation=None, expression_frames=None):
    bone_frames = []
    for index, pos in enumerate(positions):
        bf = positions_to_frames(index,pos, head_rotation)
        bone_frames.extend(bf)
    writer = VmdWriter()
    writer.write_vmd_file(vmd_file, bone_frames)

def pos2vmd(pose_3d, path, head_rotation=None, expression_frames=None):
    #for index, pose in enumerate(pose_3d):
    print("\nGenerating vmd...")
    positions = convert_position(pose_3d)
    vmd_file=path+"output_"+str(0)+".vmd"
    position_list_to_vmd(positions, vmd_file, head_rotation, expression_frames)
    print("Generating vmd successfully!")
    

