import numpy as np
from app.services.ergonomic_model import get_risk_level, get_action_level

def calculate_reba_statistics(results):
    """Calculate statistics for REBA scores"""
    reba_scores = [result['reba_score'] for result in results]
    
    return {
        'min': float(np.min(reba_scores)),
        'max': float(np.max(reba_scores)),
        'avg': float(np.mean(reba_scores)),
        'median': float(np.median(reba_scores)),
        'std': float(np.std(reba_scores))
    }

def calculate_component_averages(results):
    """Calculate average component scores"""
    component_types = ['trunk', 'neck', 'upper_arm', 'lower_arm']  # Removed leg as per expert
    
    averages = {}
    for component in component_types:
        scores = [result['component_scores'][component] for result in results]
        averages[component] = float(np.mean(scores))
    
    return averages

def calculate_angle_statistics(results):
    """Calculate statistics for joint angles"""
    angle_types = [
        'neck', 'waist', 'left_upper_arm', 'right_upper_arm',
        'left_lower_arm', 'right_lower_arm', 'left_leg', 'right_leg'
    ]
    
    angle_stats = {}
    for angle_type in angle_types:
        angles = [result['angle_values'][angle_type] for result in results]
        angle_stats[angle_type] = {
            'min': float(np.min(angles)),
            'max': float(np.max(angles)),
            'avg': float(np.mean(angles)),
            'std': float(np.std(angles))
        }
    
    return angle_stats

def identify_high_risk_periods(results, threshold=7.0, min_duration=3):
    """Identify periods of high ergonomic risk"""
    if len(results) < 2:
        return []
        
    # Sort results by frame number
    sorted_results = sorted(results, key=lambda x: x['frame'])
    
    # Find continuous sequences above threshold
    high_risk_periods = []
    current_period = None
    
    for i, result in enumerate(sorted_results):
        if result['reba_score'] >= threshold:
            if current_period is None:
                current_period = {
                    'start_frame': result['frame'],
                    'scores': [result['reba_score']]
                }
            else:
                current_period['scores'].append(result['reba_score'])
        else:
            if current_period is not None:
                current_period['end_frame'] = sorted_results[i-1]['frame']
                # Check if period is long enough
                if len(current_period['scores']) >= min_duration:
                    current_period['avg_score'] = float(np.mean(current_period['scores']))
                    high_risk_periods.append(current_period)
                current_period = None
    
    # Handle the case where the video ends during a high risk period
    if current_period is not None:
        current_period['end_frame'] = sorted_results[-1]['frame']
        if len(current_period['scores']) >= min_duration:
            current_period['avg_score'] = float(np.mean(current_period['scores']))
            high_risk_periods.append(current_period)
    
    return high_risk_periods

def generate_recommendations(results):
    """Generate simple angle-based recommendations in Indonesian"""
    reba_scores = [result['reba_score'] for result in results]
    avg_reba = np.mean(reba_scores)
    
    component_averages = calculate_component_averages(results)
    
    # Calculate average angles for specific recommendations
    angles = {}
    for result in results:
        for angle_type, angle_value in result['angle_values'].items():
            if angle_type not in angles:
                angles[angle_type] = []
            angles[angle_type].append(angle_value)
    
    avg_angles = {k: np.mean(v) for k, v in angles.items()}
    
    recommendations = []
    
    # Trunk recommendations based on actual waist angle
    if component_averages['trunk'] >= 3:
        waist_angle = avg_angles.get('waist', 90)
        if waist_angle > 105:  # Forward lean
            recommendations.append("Luruskan punggung, jangan terlalu membungkuk ke depan.")
        elif waist_angle < 85:  # Backward lean  
            recommendations.append("Duduk lebih tegak, jangan terlalu bersandar ke belakang.")
        else:
            recommendations.append("Perbaiki posisi duduk agar punggung lebih lurus.")
    
    # Neck recommendations based on actual neck angle
    if component_averages['neck'] >= 2:
        neck_angle = avg_angles.get('neck', 0)
        if neck_angle > 20:
            recommendations.append("Angkat kepala, jangan terlalu menunduk.")
        else:
            recommendations.append("Atur posisi kepala agar lebih tegak.")
    
    # Upper arm recommendations based on actual angles
    if component_averages['upper_arm'] >= 3:
        left_upper = avg_angles.get('left_upper_arm', 0)
        right_upper = avg_angles.get('right_upper_arm', 0)
        max_upper = max(abs(left_upper), abs(right_upper))
        
        if max_upper > 45:
            recommendations.append("Turunkan posisi lengan atas, jangan terlalu terangkat.")
        else:
            recommendations.append("Atur posisi lengan atas agar lebih nyaman.")
    
    # Lower arm recommendations based on actual angles
    if component_averages['lower_arm'] >= 2:
        left_lower = avg_angles.get('left_lower_arm', 90)
        right_lower = avg_angles.get('right_lower_arm', 90)
        
        if left_lower < 60 or right_lower < 60:
            recommendations.append("Buka siku lebih lebar, jangan terlalu menekuk.")
        elif left_lower > 100 or right_lower > 100:
            recommendations.append("Tekuk siku lebih dalam, jangan terlalu lurus.")
        else:
            recommendations.append("Atur sudut siku sekitar 90 derajat.")
    
    # General recommendations based on overall REBA score
    if avg_reba <= 3:
        if not recommendations:  # Only add if no specific recommendations
            recommendations.append("Postur sudah cukup baik, pertahankan posisi ini.")
    elif avg_reba <= 7:
        if len(recommendations) == 0:
            recommendations.append("Perbaiki postur duduk untuk mengurangi risiko.")
        recommendations.append("Sesekali ubah posisi untuk mengurangi kelelahan.")
    else:
        recommendations.append("Segera perbaiki postur duduk karena berisiko tinggi.")
        recommendations.append("Istirahat sejenak dan atur ulang posisi duduk.")
    
    return recommendations

def summarize_results(results):
    """
    Generate a comprehensive summary of ergonomic analysis results
    
    Args:
        results: List of frame-by-frame result dictionaries
        
    Returns:
        dict: Summary statistics and recommendations in Indonesian
    """
    if not results:
        return {"error": "Tidak ada hasil untuk dianalisis"}
    
    # Calculate REBA score statistics
    reba_stats = calculate_reba_statistics(results)
    
    # Calculate average component scores (excluding legs as per expert)
    avg_component_scores = calculate_component_averages(results)
    
    # Calculate angle statistics
    angle_stats = calculate_angle_statistics(results)
    
    # Identify high risk periods
    high_risk_periods = identify_high_risk_periods(results)
    
    # Generate recommendations in Indonesian
    recommendations = generate_recommendations(results)
    
    # Determine overall risk level
    risk_level = get_risk_level(reba_stats['avg'])
    action_level, action_text = get_action_level(reba_stats['avg'])
    
    # Create summary
    summary = {
        "avg_reba_score": reba_stats['avg'],
        "reba_statistics": reba_stats,
        "risk_level": risk_level,
        "action_level": action_level,
        "action_text": action_text,
        "avg_component_scores": avg_component_scores,
        "angle_statistics": angle_stats,
        "high_risk_periods_count": len(high_risk_periods),
        "high_risk_periods": high_risk_periods,
        "recommendations": recommendations
    }
    
    return summary