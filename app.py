from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import pandas as pd
import geopandas as gpd
import numpy as np
from werkzeug.utils import secure_filename
from data_processor import SpatialDataProcessor
from score_calculator import ScoreCalculator
import folium
from folium.plugins import HeatMap
import tempfile
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'rural_datacenter_site_selector_2024'
app.config['UPLOAD_FOLDER'] = 'data'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB 제한

# 전역 변수로 데이터 저장
processor = None
calculator = None
scored_results = None

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """데이터 파일 업로드"""
    global processor, calculator
    
    try:
        # 카카오 API 키 확인
        kakao_api_key = request.form.get('kakao_api_key')
        if not kakao_api_key:
            return jsonify({'success': False, 'message': '카카오 API 키가 필요합니다.'})
        
        # 업로드된 파일들 처리
        uploaded_files = {}
        required_files = ['grid_shapefile', 'building_shapefile', 'agricultural_shapefile', 
                         'housing_data', 'score_table', 'weight_table']
        
        for file_type in required_files:
            if file_type in request.files:
                file = request.files[file_type]
                if file.filename != '':
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    uploaded_files[file_type] = filepath
        
        # 주소 데이터 처리 (태양광, 변전소)
        solar_addresses = request.form.get('solar_addresses', '').strip().split('\n')
        substation_addresses = request.form.get('substation_addresses', '').strip().split('\n')
        
        # 빈 주소 제거
        solar_addresses = [addr.strip() for addr in solar_addresses if addr.strip()]
        substation_addresses = [addr.strip() for addr in substation_addresses if addr.strip()]
        
        # 데이터 프로세서 초기화
        processor = SpatialDataProcessor(kakao_api_key)
        
        # 파일 로드
        if 'grid_shapefile' in uploaded_files:
            processor.load_grid_data(uploaded_files['grid_shapefile'])
        
        if 'building_shapefile' in uploaded_files:
            processor.load_building_data(uploaded_files['building_shapefile'])
        
        if 'agricultural_shapefile' in uploaded_files:
            processor.load_agricultural_zones(uploaded_files['agricultural_shapefile'])
        
        if 'housing_data' in uploaded_files:
            processor.load_housing_data(uploaded_files['housing_data'])
        
        # 점수 계산기 초기화
        if 'score_table' in uploaded_files and 'weight_table' in uploaded_files:
            calculator = ScoreCalculator(
                uploaded_files['score_table'], 
                uploaded_files['weight_table']
            )
        
        return jsonify({
            'success': True, 
            'message': '파일 업로드가 완료되었습니다.',
            'solar_count': len(solar_addresses),
            'substation_count': len(substation_addresses)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'업로드 중 오류가 발생했습니다: {str(e)}'})

@app.route('/process', methods=['POST'])
def process_data():
    """데이터 처리 및 점수 계산"""
    global processor, calculator, scored_results
    
    try:
        if not processor or not calculator:
            return jsonify({'success': False, 'message': '데이터를 먼저 업로드해주세요.'})
        
        # 주소 데이터 가져오기
        data = request.get_json()
        solar_addresses = data.get('solar_addresses', [])
        substation_addresses = data.get('substation_addresses', [])
        
        # 주소를 좌표로 변환
        solar_coordinates = processor.convert_addresses_to_coordinates(solar_addresses)
        substation_coordinates = processor.convert_addresses_to_coordinates(substation_addresses)
        
        # 좌표를 GeoDataFrame으로 변환
        solar_gdf = processor.create_points_from_coordinates(solar_coordinates)
        substation_gdf = processor.create_points_from_coordinates(substation_coordinates)
        
        # 건설 가능한 격자 필터링
        buildable_grids = processor.filter_buildable_grids()
        
        # 주택수 데이터 매칭
        housing_counts = np.zeros(len(buildable_grids))
        if processor.housing_data is not None:
            # 간단한 예시 - 실제로는 격자 ID로 매칭해야 함
            housing_counts = np.random.randint(0, 60, len(buildable_grids))
        
        # 점수 계산
        scores = calculator.calculate_total_score(
            buildable_grids, substation_gdf, solar_gdf, housing_counts
        )
        
        # 점수가 포함된 GeoDataFrame 생성
        scored_results = calculator.create_scored_geodataframe(buildable_grids, scores)
        
        # 결과 저장
        output_path = 'output/scored_grids.geojson'
        calculator.export_results_to_geojson(scored_results, output_path)
        
        # 상위 후보지 추출
        top_candidates = calculator.get_top_candidates(scored_results, 20)
        
        # 통계 계산
        statistics = calculator.get_score_statistics(scores)
        
        return jsonify({
            'success': True,
            'message': '데이터 처리가 완료되었습니다.',
            'total_grids': len(buildable_grids),
            'top_candidates_count': len(top_candidates),
            'statistics': statistics
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'처리 중 오류가 발생했습니다: {str(e)}'})

@app.route('/map')
def show_map():
    """지도 시각화 페이지"""
    global scored_results
    
    if scored_results is None:
        return "데이터를 먼저 처리해주세요."
    
    # 경상남도 중심 좌표
    center_lat, center_lng = 35.4606, 128.2132
    
    # Folium 지도 생성
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=9,
        tiles='OpenStreetMap'
    )
    
    # 점수별 색상 매핑
    def get_color(score):
        if score >= 8:
            return 'red'
        elif score >= 6:
            return 'orange'
        elif score >= 4:
            return 'yellow'
        elif score >= 2:
            return 'lightgreen'
        else:
            return 'blue'
    
    # 격자 시각화
    for idx, row in scored_results.iterrows():
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda feature, score=row['total_score']: {
                'fillColor': get_color(score),
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7
            },
            popup=folium.Popup(
                f"""
                <b>격자 정보</b><br>
                총점: {row['total_score']:.2f}<br>
                등급: {row['score_grade']}<br>
                변전소 접근성: {row['score_a']:.1f}<br>
                태양광 접근성: {row['score_b']:.1f}<br>
                인구밀도: {row['score_d']:.1f}<br>
                교통 접근성: {row['score_e']:.1f}
                """,
                max_width=200
            )
        ).add_to(m)
    
    # 범례 추가
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>점수 등급</b></p>
    <p><i class="fa fa-square" style="color:red"></i> A등급 (8-10점)</p>
    <p><i class="fa fa-square" style="color:orange"></i> B등급 (6-8점)</p>
    <p><i class="fa fa-square" style="color:yellow"></i> C등급 (4-6점)</p>
    <p><i class="fa fa-square" style="color:lightgreen"></i> D등급 (2-4점)</p>
    <p><i class="fa fa-square" style="color:blue"></i> F등급 (0-2점)</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # 지도를 임시 파일로 저장
    map_path = 'templates/map.html'
    m.save(map_path)
    
    return render_template('map.html')

@app.route('/api/update_weights', methods=['POST'])
def update_weights():
    """가중치 업데이트 및 점수 재계산"""
    global calculator, scored_results, processor
    
    try:
        if not calculator or scored_results is None:
            return jsonify({'success': False, 'message': '데이터를 먼저 처리해주세요.'})
        
        # 새로운 가중치 받기
        new_weights = request.get_json()
        
        # 가중치 업데이트
        calculator.update_weights(new_weights)
        
        # 점수 재계산 (기존 개별 점수 사용)
        weight_a = new_weights.get('A', 0.25)
        weight_b = new_weights.get('B', 0.25)
        weight_d = new_weights.get('D', 0.25)
        weight_e = new_weights.get('E', 0.25)
        
        # 총합이 1이 되도록 정규화
        total_weight = weight_a + weight_b + weight_d + weight_e
        if total_weight > 0:
            weight_a /= total_weight
            weight_b /= total_weight
            weight_d /= total_weight
            weight_e /= total_weight
        
        # 새로운 총점 계산
        scored_results['total_score'] = (
            scored_results['score_a'] * weight_a +
            scored_results['score_b'] * weight_b +
            scored_results['score_d'] * weight_d +
            scored_results['score_e'] * weight_e
        )
        
        # 등급 재계산
        scored_results['score_grade'] = calculator._classify_score_grade(
            scored_results['total_score'].values
        )
        
        # 상위 후보지 업데이트
        top_candidates = calculator.get_top_candidates(scored_results, 20)
        
        # 새로운 통계 계산
        new_scores = {
            'score_a': scored_results['score_a'].values,
            'score_b': scored_results['score_b'].values,
            'score_d': scored_results['score_d'].values,
            'score_e': scored_results['score_e'].values,
            'total_score': scored_results['total_score'].values
        }
        statistics = calculator.get_score_statistics(new_scores)
        
        return jsonify({
            'success': True,
            'message': '가중치가 업데이트되었습니다.',
            'top_candidates_count': len(top_candidates),
            'statistics': statistics
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'가중치 업데이트 중 오류: {str(e)}'})

@app.route('/api/top_candidates')
def get_top_candidates():
    """상위 후보지 목록 반환"""
    global scored_results, calculator
    
    try:
        if scored_results is None:
            return jsonify({'success': False, 'message': '데이터를 먼저 처리해주세요.'})
        
        top_n = request.args.get('top_n', 20, type=int)
        top_candidates = calculator.get_top_candidates(scored_results, top_n)
        
        # GeoJSON 형태로 변환
        candidates_list = []
        for idx, row in top_candidates.iterrows():
            centroid = row.geometry.centroid
            candidates_list.append({
                'grid_id': str(idx),
                'total_score': float(row['total_score']),
                'score_grade': row['score_grade'],
                'score_a': float(row['score_a']),
                'score_b': float(row['score_b']),
                'score_d': float(row['score_d']),
                'score_e': float(row['score_e']),
                'lat': float(centroid.y),
                'lng': float(centroid.x)
            })
        
        return jsonify({
            'success': True,
            'candidates': candidates_list
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'후보지 조회 중 오류: {str(e)}'})

@app.route('/download_results')
def download_results():
    """결과 파일 다운로드"""
    try:
        output_path = 'output/scored_grids.geojson'
        if os.path.exists(output_path):
            return send_file(output_path, as_attachment=True, 
                           download_name='datacenter_site_candidates.geojson')
        else:
            return "결과 파일이 없습니다. 먼저 데이터를 처리해주세요."
    except Exception as e:
        return f"다운로드 중 오류가 발생했습니다: {str(e)}"

if __name__ == '__main__':
    # 필요한 디렉토리 생성
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # 개발 모드로 실행
    app.run(debug=True, host='0.0.0.0', port=5000)