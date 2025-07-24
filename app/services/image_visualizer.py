import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

# Constants
KEYPOINT_THRESHOLD = 0.3
KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3,
    'right_ear': 4, 'left_shoulder': 5, 'right_shoulder': 6, 
    'left_elbow': 7, 'right_elbow': 8, 'left_wrist': 9,
    'right_wrist': 10, 'left_hip': 11, 'right_hip': 12, 
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

# FIXED: Edge colors for visualization - NOW INCLUDING LEGS
KEYPOINT_EDGE_INDS_TO_COLOR = {
    # Head connections
    (5, 3): 'r', (6, 4): 'r', 
    # Arms
    (5, 7): 'b', (7, 9): 'b', (6, 8): 'b', (8, 10): 'b',
    # Torso
    (5, 6): 'b', (5, 11): 'orange', (6, 12): 'orange', 
    # Hip connection
    (11, 12): 'orange',
    # FIXED: Legs - These were missing!
    (11, 13): 'purple', (13, 15): 'purple',  # Left leg
    (12, 14): 'purple', (14, 16): 'purple'   # Right leg
}

# Risk level colors - ADDED LEG COLORS
RISK_COLORS = {
    'trunk': {
        1: '#00FF00', 2: '#FFFF00', 3: '#FFA500', 
        4: '#FF0000', 5: '#FF0000', 6: '#FF0000'
    },
    'neck': {
        1: '#00FF00', 2: '#FFFF00', 3: '#FFA500', 4: '#FF0000'
    },
    'upper_arm': {
        1: '#00FF00', 2: '#FFFF00', 3: '#FFA500', 4: '#FF0000'
    },
    'lower_arm': {
        1: '#00FF00', 2: '#FF0000'
    },
    # ADDED: Leg colors (using similar scheme to arms)
    'leg': {
        1: '#00FF00', 2: '#FFFF00', 3: '#FFA500', 4: '#FF0000'
    }
}

def _keypoints_and_edges_for_display(keypoints_with_scores, height, width, keypoint_threshold=KEYPOINT_THRESHOLD):
    """Prepare keypoints and edges for visualization"""
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape

    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack([width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
        kpts_above_thresh_absolute = kpts_absolute_xy[kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)

        for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if kpts_scores[edge_pair[0]] > keypoint_threshold and kpts_scores[edge_pair[1]] > keypoint_threshold:
                x_start, y_start = kpts_absolute_xy[edge_pair[0]]
                x_end, y_end = kpts_absolute_xy[edge_pair[1]]
                line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                keypoint_edges_all.append(line_seg)
                edge_colors.append(color)

    keypoints_xy = np.concatenate(keypoints_all, axis=0) if keypoints_all else np.zeros((0, 2))
    edges_xy = np.stack(keypoint_edges_all, axis=0) if keypoint_edges_all else np.zeros((0, 2, 2))

    return keypoints_xy, edges_xy, edge_colors

def check_imputed_joints(angle_values):
    """Check which joints were imputed based on angle values"""
    imputed_joints = {}
    
    if 'imputed_angles' in angle_values:
        imputed_info = angle_values['imputed_angles']
        imputed_joints = {
            'neck': imputed_info.get('neck', False),
            'trunk': imputed_info.get('waist', False),
            'left_upper_arm': imputed_info.get('left_upper_arm', False),
            'right_upper_arm': imputed_info.get('right_upper_arm', False),
            'left_lower_arm': imputed_info.get('left_lower_arm', False),
            'right_lower_arm': imputed_info.get('right_lower_arm', False),
            'left_leg': imputed_info.get('left_leg', False),
            'right_leg': imputed_info.get('right_leg', False)
        }
    else:
        # Fallback: check for standard angle values that indicate imputation
        imputed_joints = {
            'neck': angle_values.get('neck', 0) == 5,
            'trunk': angle_values.get('waist', 0) == 110,
            'left_upper_arm': angle_values.get('left_upper_arm', 0) == 0,
            'right_upper_arm': angle_values.get('right_upper_arm', 0) == 0,
            'left_lower_arm': angle_values.get('left_lower_arm', 0) == 90,
            'right_lower_arm': angle_values.get('right_lower_arm', 0) == 90,
            'left_leg': angle_values.get('left_leg', 0) == 90,
            'right_leg': angle_values.get('right_leg', 0) == 90
        }
    
    return imputed_joints

def generate_pose_visualization(image, keypoints_with_scores, risk_scores, original_image=None, 
                               flip_applied=False, crop_region=None, output_image_height=None,
                               angle_values=None):
    """Draws keypoint predictions with risk level coloring and imputation indicators"""
   
    vis_image = image

    height, width, _ = vis_image.shape
    aspect_ratio = float(width) / height
    
    # Create figure with explicit backend
    plt.ioff()  # Turn off interactive mode
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')

    im = ax.imshow(vis_image)
    
    # Check for imputed joints
    imputed_joints = {}
    if angle_values:
        imputed_joints = check_imputed_joints(angle_values)
    
    # FIXED: Define the body segments and their risk types - LEGS USE WHITE (NOT ASSESSED)
    risk_segments = {
        ((3, 5), (4, 6)): ('neck', imputed_joints.get('neck', False)),  
        ((5, 11), (6, 12)): ('trunk', imputed_joints.get('trunk', False)),
        ((5, 7),): ('upper_arm', imputed_joints.get('left_upper_arm', False)),
        ((6, 8),): ('upper_arm', imputed_joints.get('right_upper_arm', False)),
        ((7, 9),): ('lower_arm', imputed_joints.get('left_lower_arm', False)),
        ((8, 10),): ('lower_arm', imputed_joints.get('right_lower_arm', False)),
        # ADDED: Leg segments - ALWAYS WHITE (not assessed per expert recommendation)
        ((11, 13),): ('leg_white', False),   # Left thigh - always white
        ((13, 15),): ('leg_white', False),   # Left shin - always white
        ((12, 14),): ('leg_white', False),   # Right thigh - always white
        ((14, 16),): ('leg_white', False)    # Right shin - always white
    }
    
    # Draw connection between shoulders
    if (keypoints_with_scores[0, 0, 5, 2] > KEYPOINT_THRESHOLD and 
        keypoints_with_scores[0, 0, 6, 2] > KEYPOINT_THRESHOLD):
        x1 = keypoints_with_scores[0, 0, 5, 1] * width
        y1 = keypoints_with_scores[0, 0, 5, 0] * height
        x2 = keypoints_with_scores[0, 0, 6, 1] * width
        y2 = keypoints_with_scores[0, 0, 6, 0] * height
        
        shoulder_line = plt.Line2D([x1, x2], [y1, y2], lw=3, color='blue', zorder=2)
        ax.add_line(shoulder_line)
    
    # ADDED: Draw connection between hips
    if (keypoints_with_scores[0, 0, 11, 2] > KEYPOINT_THRESHOLD and 
        keypoints_with_scores[0, 0, 12, 2] > KEYPOINT_THRESHOLD):
        x1 = keypoints_with_scores[0, 0, 11, 1] * width
        y1 = keypoints_with_scores[0, 0, 11, 0] * height
        x2 = keypoints_with_scores[0, 0, 12, 1] * width
        y2 = keypoints_with_scores[0, 0, 12, 0] * height
        
        hip_line = plt.Line2D([x1, x2], [y1, y2], lw=3, color='orange', zorder=2)
        ax.add_line(hip_line)
    
    # Draw edges with risk-based coloring or white for imputed/legs
    for segment_pairs, (risk_type, is_imputed) in risk_segments.items():
        if is_imputed:
            color = 'white'
        elif risk_type == 'leg_white':
            color = 'white'  # Legs always white (not assessed)
        else:
            risk_level = risk_scores.get(f'{risk_type}_score', 1)
            color = RISK_COLORS[risk_type].get(risk_level, 'gray')
        
        for edge_pair in segment_pairs:
            idx1, idx2 = edge_pair
            
            if (keypoints_with_scores[0, 0, idx1, 2] > KEYPOINT_THRESHOLD and 
                keypoints_with_scores[0, 0, idx2, 2] > KEYPOINT_THRESHOLD):
                
                x1 = keypoints_with_scores[0, 0, idx1, 1] * width
                y1 = keypoints_with_scores[0, 0, idx1, 0] * height
                x2 = keypoints_with_scores[0, 0, idx2, 1] * width
                y2 = keypoints_with_scores[0, 0, idx2, 0] * height
                
                line = plt.Line2D([x1, x2], [y1, y2], lw=4, color=color, zorder=2)
                ax.add_line(line)
    
    # Draw keypoints
    valid_keypoints = []
    for i, score in enumerate(keypoints_with_scores[0, 0, :, 2]):
        if score > KEYPOINT_THRESHOLD:
            x = keypoints_with_scores[0, 0, i, 1] * width
            y = keypoints_with_scores[0, 0, i, 0] * height
            valid_keypoints.append((x, y))
    
    if valid_keypoints:
        keypoints_x, keypoints_y = zip(*valid_keypoints)
        ax.scatter(keypoints_x, keypoints_y, s=60, color='white', edgecolor='black', zorder=3)
    
    # Add predicted REBA score text
    if 'reba_score' in risk_scores:
        reba_score = risk_scores['reba_score']
        if reba_score <= 2:
            reba_color = 'darkgreen'
        elif reba_score <= 4:
            reba_color = 'orange'
        elif reba_score <= 7:
            reba_color = 'darkorange'
        else:
            reba_color = 'darkred'
            
        text = f"REBA Score: {reba_score:.1f}"
        ax.text(width * 0.05, height * 0.05, text, fontsize=18, 
                color='white', bbox=dict(facecolor=reba_color, alpha=0.8))
    
    # UPDATED: Add risk legend - NOW INCLUDING LEGS
    legend_y = height * 0.95
    ax.text(width * 0.05, legend_y, "Skor REBA per Bagian:", fontsize=14, color='black', 
            bbox=dict(facecolor='white', alpha=0.8))
    
    legend_items = [
        ("Leher", risk_scores.get('neck_score', 1), 'neck'),
        ("Batang Tubuh", risk_scores.get('trunk_score', 1), 'trunk'),
        ("Lengan Atas", risk_scores.get('upper_arm_score', 1), 'upper_arm'),
        ("Lengan Bawah", risk_scores.get('lower_arm_score', 1), 'lower_arm'),
        ("Kaki", "Tidak Dinilai", 'leg_white')  
    ]
    
    y_offset = 0.05
    for i, (name, level, risk_type) in enumerate(legend_items):
        y_pos = legend_y - (i+1) * height * y_offset
        if risk_type == 'leg_white':
            color = 'white'  # Legs always white
        else:
            color = RISK_COLORS[risk_type].get(level, 'gray')
        ax.text(width * 0.07, y_pos, f"• {name}: {level}", fontsize=12, color='black',
                bbox=dict(facecolor=color, alpha=0.7, edgecolor='black' if risk_type == 'leg_white' else None))
    
    # Add imputation indicator
    if any(imputed_joints.values()):
        y_pos = legend_y - len(legend_items) * height * y_offset - height * 0.02
        ax.text(width * 0.07, y_pos, "• Putih: Data Diestimasi", fontsize=12, color='black',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
    
    # Convert figure to image (ROBUST VERSION)
    fig.canvas.draw()
    
    try:
        # Method 1: Try the standard approach
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    except AttributeError:
        try:
            # Method 2: Alternative for older matplotlib
            renderer = fig.canvas.get_renderer()
            raw_data = renderer.tostring_rgb()
            image_from_plot = np.frombuffer(raw_data, dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except:
            try:
                # Method 3: Use print_to_buffer
                buf, (w, h) = fig.canvas.print_to_buffer()
                image_from_plot = np.frombuffer(buf, dtype=np.uint8)
                if len(image_from_plot.shape) == 1:
                    image_from_plot = image_from_plot.reshape((h, w, 4))[:, :, :3]  # RGBA to RGB
            except:
                # Method 4: Fallback - create a blank image
                print("Warning: Could not extract visualization, creating blank image")
                image_from_plot = np.ones((height, width, 3), dtype=np.uint8) * 128
    
    plt.close(fig)
    plt.ion()  # Turn interactive mode back on
    
    # Resize if requested
    if output_image_height is not None:
        output_image_width = int(output_image_height / height * width)
        image_from_plot = cv2.resize(image_from_plot, dsize=(output_image_width, output_image_height),
                                     interpolation=cv2.INTER_CUBIC)
    
    return image_from_plot