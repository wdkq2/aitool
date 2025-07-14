import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, List, Tuple
import json

class ScoreCalculator:
    """격자별 점수 계산을 담당하는 클래스"""
    
    def __init__(self, score_table_path: str, weight_table_path: str):
        self.score_table = pd.read_csv(score_table_path)
        self.weight_table = pd.read_csv(weight_table_path)
        self.weights = self._load_weights()
        
    def _load_weights(self) -> Dict[str, float]:
        """가중치 테이블을 딕셔너리로 변환합니다."""
        weights = {}
        for _, row in self.weight_table.iterrows():
            weights[row['category']] = row['weight']
        return weights
    
    def update_weights(self, new_weights: Dict[str, float]):
        """가중치를 업데이트합니다."""
        self.weights.update(new_weights)
    
    def calculate_category_a_score(self, 
                                 grid_gdf: gpd.GeoDataFrame, 
                                 substation_gdf: gpd.GeoDataFrame) -> np.ndarray:
        """A항: 변전소 접근성 점수 계산"""
        distances = []
        
        for grid_geom in grid_gdf.geometry:
            grid_centroid = grid_geom.centroid
            min_distance = float('inf')
            
            for substation_geom in substation_gdf.geometry:
                # 거리 계산 (단위: km)
                distance = grid_centroid.distance(substation_geom) * 111  # 대략적인 위경도-km 변환
                min_distance = min(min_distance, distance)
            
            distances.append(min_distance)
        
        distances = np.array(distances)
        
        # 점수표에 따른 점수 매기기
        scores = np.zeros_like(distances)
        scores[distances <= 1] = 10    # 1km 이내
        scores[(distances > 1) & (distances <= 3)] = 8   # 1-3km
        scores[(distances > 3) & (distances <= 5)] = 6   # 3-5km
        scores[(distances > 5) & (distances <= 10)] = 4  # 5-10km
        scores[distances > 10] = 2     # 10km 초과
        
        return scores
    
    def calculate_category_b_score(self, 
                                 grid_gdf: gpd.GeoDataFrame, 
                                 solar_gdf: gpd.GeoDataFrame) -> np.ndarray:
        """B항: 태양광 발전소 접근성 점수 계산"""
        distances = []
        
        for grid_geom in grid_gdf.geometry:
            grid_centroid = grid_geom.centroid
            min_distance = float('inf')
            
            for solar_geom in solar_gdf.geometry:
                # 거리 계산 (단위: km)
                distance = grid_centroid.distance(solar_geom) * 111
                min_distance = min(min_distance, distance)
            
            distances.append(min_distance)
        
        distances = np.array(distances)
        
        # 점수표에 따른 점수 매기기
        scores = np.zeros_like(distances)
        scores[distances <= 2] = 10    # 2km 이내
        scores[(distances > 2) & (distances <= 5)] = 8   # 2-5km
        scores[(distances > 5) & (distances <= 10)] = 6  # 5-10km
        scores[(distances > 10) & (distances <= 15)] = 4 # 10-15km
        scores[distances > 15] = 2     # 15km 초과
        
        return scores
    
    def calculate_category_d_score(self, housing_counts: np.ndarray) -> np.ndarray:
        """D항: 농촌인구밀도 점수 계산 (격자별 주택수 기준)"""
        scores = np.zeros_like(housing_counts, dtype=float)
        
        # 주택수에 따른 점수 매기기
        scores[housing_counts >= 50] = 10    # 50가구 이상
        scores[(housing_counts >= 30) & (housing_counts < 50)] = 8   # 30-49가구
        scores[(housing_counts >= 20) & (housing_counts < 30)] = 6   # 20-29가구
        scores[(housing_counts >= 10) & (housing_counts < 20)] = 4   # 10-19가구
        scores[housing_counts < 10] = 2      # 10가구 미만
        
        return scores
    
    def calculate_category_e_score(self, 
                                 grid_gdf: gpd.GeoDataFrame, 
                                 road_network_gdf: gpd.GeoDataFrame = None) -> np.ndarray:
        """E항: 교통 접근성 점수 계산"""
        # 도로 네트워크 데이터가 없는 경우 기본값 사용
        if road_network_gdf is None:
            # 모든 격자에 중간 점수 부여 (실제 구현 시 도로 데이터 필요)
            return np.full(len(grid_gdf), 6.0)
        
        distances = []
        
        for grid_geom in grid_gdf.geometry:
            grid_centroid = grid_geom.centroid
            min_distance = float('inf')
            
            for road_geom in road_network_gdf.geometry:
                distance = grid_centroid.distance(road_geom) * 111
                min_distance = min(min_distance, distance)
            
            distances.append(min_distance)
        
        distances = np.array(distances)
        
        # 도로와의 거리에 따른 점수 매기기
        scores = np.zeros_like(distances)
        scores[distances <= 0.5] = 10   # 500m 이내
        scores[(distances > 0.5) & (distances <= 1)] = 8    # 0.5-1km
        scores[(distances > 1) & (distances <= 2)] = 6      # 1-2km
        scores[(distances > 2) & (distances <= 5)] = 4      # 2-5km
        scores[distances > 5] = 2       # 5km 초과
        
        return scores
    
    def calculate_total_score(self, 
                            grid_gdf: gpd.GeoDataFrame,
                            substation_gdf: gpd.GeoDataFrame,
                            solar_gdf: gpd.GeoDataFrame,
                            housing_counts: np.ndarray,
                            road_network_gdf: gpd.GeoDataFrame = None) -> Dict[str, np.ndarray]:
        """전체 점수를 계산합니다."""
        
        # 각 카테고리별 점수 계산
        score_a = self.calculate_category_a_score(grid_gdf, substation_gdf)
        score_b = self.calculate_category_b_score(grid_gdf, solar_gdf)
        # score_c는 용수 공급원 데이터가 준비되면 추가
        score_d = self.calculate_category_d_score(housing_counts)
        score_e = self.calculate_category_e_score(grid_gdf, road_network_gdf)
        
        # 가중치 적용
        weight_a = self.weights.get('A', 0.25)  # 변전소 접근성
        weight_b = self.weights.get('B', 0.25)  # 태양광 접근성
        weight_c = self.weights.get('C', 0.0)   # 용수 공급원 (데이터 없음)
        weight_d = self.weights.get('D', 0.25)  # 농촌인구밀도
        weight_e = self.weights.get('E', 0.25)  # 교통 접근성
        
        # 최종 점수 계산
        total_score = (score_a * weight_a + 
                      score_b * weight_b + 
                      score_d * weight_d + 
                      score_e * weight_e)
        
        return {
            'score_a': score_a,
            'score_b': score_b,
            'score_d': score_d,
            'score_e': score_e,
            'total_score': total_score
        }
    
    def create_scored_geodataframe(self, 
                                 grid_gdf: gpd.GeoDataFrame,
                                 scores: Dict[str, np.ndarray]) -> gpd.GeoDataFrame:
        """점수가 포함된 GeoDataFrame을 생성합니다."""
        result_gdf = grid_gdf.copy()
        
        for score_type, score_values in scores.items():
            result_gdf[score_type] = score_values
        
        # 점수 등급 추가
        result_gdf['score_grade'] = self._classify_score_grade(scores['total_score'])
        
        return result_gdf
    
    def _classify_score_grade(self, scores: np.ndarray) -> List[str]:
        """점수를 등급으로 분류합니다."""
        grades = []
        for score in scores:
            if score >= 8:
                grades.append('A')
            elif score >= 6:
                grades.append('B')
            elif score >= 4:
                grades.append('C')
            elif score >= 2:
                grades.append('D')
            else:
                grades.append('F')
        return grades
    
    def get_top_candidates(self, 
                         scored_gdf: gpd.GeoDataFrame, 
                         top_n: int = 20) -> gpd.GeoDataFrame:
        """상위 N개 후보지를 반환합니다."""
        return scored_gdf.nlargest(top_n, 'total_score')
    
    def export_results_to_geojson(self, 
                                scored_gdf: gpd.GeoDataFrame, 
                                output_path: str):
        """결과를 GeoJSON 파일로 저장합니다."""
        # CRS를 WGS84로 변환
        if scored_gdf.crs != 'EPSG:4326':
            scored_gdf = scored_gdf.to_crs('EPSG:4326')
        
        scored_gdf.to_file(output_path, driver='GeoJSON')
        print(f"결과가 {output_path}에 저장되었습니다.")
    
    def get_score_statistics(self, scores: Dict[str, np.ndarray]) -> Dict:
        """점수 통계를 계산합니다."""
        stats = {}
        
        for score_type, score_values in scores.items():
            stats[score_type] = {
                'mean': float(np.mean(score_values)),
                'std': float(np.std(score_values)),
                'min': float(np.min(score_values)),
                'max': float(np.max(score_values)),
                'median': float(np.median(score_values))
            }
        
        return stats