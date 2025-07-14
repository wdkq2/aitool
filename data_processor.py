import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import time
from shapely.geometry import Point, Polygon
from typing import Dict, List, Tuple, Optional
import json
import os

class SpatialDataProcessor:
    """공간 데이터 처리를 담당하는 클래스"""
    
    def __init__(self, kakao_api_key: str):
        self.kakao_api_key = kakao_api_key
        self.grid_gdf = None
        self.buildings_gdf = None
        self.agricultural_zones_gdf = None
        self.housing_data = None
        
    def load_grid_data(self, grid_shapefile_path: str) -> gpd.GeoDataFrame:
        """100m 격자 지도 데이터를 로드합니다."""
        try:
            self.grid_gdf = gpd.read_file(grid_shapefile_path)
            print(f"격자 데이터 로드 완료: {len(self.grid_gdf)}개 격자")
            return self.grid_gdf
        except Exception as e:
            print(f"격자 데이터 로드 실패: {e}")
            return None
    
    def load_building_data(self, building_shapefile_path: str) -> gpd.GeoDataFrame:
        """건축물 정보 데이터를 로드합니다."""
        try:
            self.buildings_gdf = gpd.read_file(building_shapefile_path)
            print(f"건축물 데이터 로드 완료: {len(self.buildings_gdf)}개 건축물")
            return self.buildings_gdf
        except Exception as e:
            print(f"건축물 데이터 로드 실패: {e}")
            return None
    
    def load_agricultural_zones(self, agricultural_shapefile_path: str) -> gpd.GeoDataFrame:
        """농업진흥지역도 데이터를 로드합니다."""
        try:
            self.agricultural_zones_gdf = gpd.read_file(agricultural_shapefile_path)
            print(f"농업진흥지역 데이터 로드 완료: {len(self.agricultural_zones_gdf)}개 구역")
            return self.agricultural_zones_gdf
        except Exception as e:
            print(f"농업진흥지역 데이터 로드 실패: {e}")
            return None
    
    def load_housing_data(self, housing_txt_path: str) -> pd.DataFrame:
        """격자별 주택수 데이터를 로드합니다."""
        try:
            self.housing_data = pd.read_csv(housing_txt_path, sep='\t')
            print(f"주택수 데이터 로드 완료: {len(self.housing_data)}개 격자")
            return self.housing_data
        except Exception as e:
            print(f"주택수 데이터 로드 실패: {e}")
            return None
    
    def address_to_coordinates(self, address: str) -> Optional[Tuple[float, float]]:
        """카카오 API를 사용하여 주소를 좌표로 변환합니다."""
        url = "https://dapi.kakao.com/v2/local/search/address.json"
        headers = {"Authorization": f"KakaoAK {self.kakao_api_key}"}
        params = {"query": address}
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data['documents']:
                lng = float(data['documents'][0]['x'])
                lat = float(data['documents'][0]['y'])
                return lng, lat
            else:
                print(f"주소를 찾을 수 없습니다: {address}")
                return None
                
        except Exception as e:
            print(f"주소 변환 실패 ({address}): {e}")
            return None
    
    def convert_addresses_to_coordinates(self, addresses: List[str]) -> List[Tuple[float, float]]:
        """여러 주소를 일괄로 좌표로 변환합니다."""
        coordinates = []
        
        for i, address in enumerate(addresses):
            if i > 0 and i % 100 == 0:  # API 호출 제한 고려
                time.sleep(1)
            
            coord = self.address_to_coordinates(address)
            if coord:
                coordinates.append(coord)
            
            print(f"진행률: {i+1}/{len(addresses)}")
        
        return coordinates
    
    def create_points_from_coordinates(self, coordinates: List[Tuple[float, float]]) -> gpd.GeoDataFrame:
        """좌표 리스트로부터 Point 지오메트리를 생성합니다."""
        points = [Point(lng, lat) for lng, lat in coordinates]
        gdf = gpd.GeoDataFrame(geometry=points, crs='EPSG:4326')
        return gdf
    
    def filter_buildable_grids(self, exclusion_types: List[str] = None) -> gpd.GeoDataFrame:
        """건설 가능한 격자만 필터링합니다."""
        if self.grid_gdf is None:
            raise ValueError("격자 데이터가 로드되지 않았습니다.")
        
        if exclusion_types is None:
            exclusion_types = ['농지', '건축물', '주거지역', '도로', '과수원']
        
        buildable_grids = self.grid_gdf.copy()
        
        # 건축물과 겹치는 격자 제외
        if self.buildings_gdf is not None:
            overlapping_indices = gpd.sjoin(
                buildable_grids, self.buildings_gdf, 
                how='inner', predicate='intersects'
            ).index
            buildable_grids = buildable_grids.drop(overlapping_indices)
            print(f"건축물 제외 후: {len(buildable_grids)}개 격자 남음")
        
        # 농업진흥지역과 겹치는 격자 제외 (선택적)
        if self.agricultural_zones_gdf is not None:
            overlapping_indices = gpd.sjoin(
                buildable_grids, self.agricultural_zones_gdf, 
                how='inner', predicate='intersects'
            ).index
            buildable_grids = buildable_grids.drop(overlapping_indices)
            print(f"농업지역 제외 후: {len(buildable_grids)}개 격자 남음")
        
        return buildable_grids
    
    def calculate_distance_to_facilities(self, 
                                       grid_gdf: gpd.GeoDataFrame, 
                                       facility_gdf: gpd.GeoDataFrame) -> np.ndarray:
        """격자와 시설 간의 최단 거리를 계산합니다."""
        distances = []
        
        for grid_geom in grid_gdf.geometry:
            grid_centroid = grid_geom.centroid
            min_distance = float('inf')
            
            for facility_geom in facility_gdf.geometry:
                distance = grid_centroid.distance(facility_geom)
                min_distance = min(min_distance, distance)
            
            distances.append(min_distance)
        
        return np.array(distances)
    
    def get_grid_info(self, grid_id: str) -> Dict:
        """특정 격자의 상세 정보를 반환합니다."""
        if self.grid_gdf is None:
            return {}
        
        grid_row = self.grid_gdf[self.grid_gdf.index == grid_id]
        if len(grid_row) == 0:
            return {}
        
        geom = grid_row.geometry.iloc[0]
        centroid = geom.centroid
        
        info = {
            'grid_id': grid_id,
            'centroid_lat': centroid.y,
            'centroid_lng': centroid.x,
            'area': geom.area,
            'bounds': geom.bounds
        }
        
        # 주택수 정보 추가
        if self.housing_data is not None:
            housing_info = self.housing_data[
                self.housing_data['grid_id'] == grid_id
            ]
            if len(housing_info) > 0:
                info['housing_count'] = housing_info['housing_count'].iloc[0]
        
        return info