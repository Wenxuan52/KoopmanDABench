import os
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta
import glob
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataConsolidatorFull:
    def __init__(self, base_path, material):
        """
        气体数据完整整合器
        """
        self.base_path = base_path
        self.material = material
        self.years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        
        # 预期的数据维度
        self.expected_lat_size = 2893
        self.expected_lon_size = 2439
        
        # 用于存储参考坐标
        self.reference_lat = None
        self.reference_lon = None
        
        logger.info(f"初始化{material}数据整合器 - 基础路径: {self.base_path}")
    
    def is_leap_year(self, year):
        """判断是否为闰年"""
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    
    def get_days_in_year(self, year):
        """获取指定年份的天数"""
        return 366 if self.is_leap_year(year) else 365
    
    def generate_date_list(self, year):
        """生成指定年份的所有日期列表"""
        start_date = datetime(year, 1, 1)
        dates = []
        
        days_in_year = self.get_days_in_year(year)
        for i in range(days_in_year):
            current_date = start_date + timedelta(days=i)
            dates.append(current_date.strftime('%Y%m%d'))
        
        return dates
    
    def find_nc_files(self, year):
        """查找指定年份的所有NetCDF文件"""
        year_path = os.path.join(self.base_path, self.material, str(year))
        
        if not os.path.exists(year_path):
            logger.warning(f"路径不存在: {year_path}")
            return {}
        
        pattern = os.path.join(year_path, f"ECHAP_{self.material}_D1K_{year}*_V1.nc")
        files = glob.glob(pattern)
        
        # 创建日期到文件路径的映射
        date_to_file = {}
        for file_path in files:
            filename = os.path.basename(file_path)
            try:
                date_part = filename.split('_')[3]
                date_to_file[date_part] = file_path
            except IndexError:
                logger.warning(f"文件名格式异常: {filename}")
                continue
        
        logger.info(f"找到 {self.material} {year} 年的文件数量: {len(date_to_file)}")
        return date_to_file
    
    def convert_to_array(self, data):
        """安全地将MaskedArray转换为numpy数组"""
        try:
            # 处理MaskedArray
            if hasattr(data, 'filled'):
                converted = data.filled(np.nan)
            else:
                converted = data
            
            # 转换为float32类型的numpy数组
            result = np.asarray(converted, dtype=np.float32)
            return result
            
        except Exception as e:
            logger.error(f"数组转换失败: {e}")
            raise
    
    def load_nc_file(self, file_path):
        """加载单个NetCDF文件"""
        try:
            with nc.Dataset(file_path, 'r') as dataset:
                # 读取原始数据
                lat_raw = dataset.variables['lat'][:]
                lon_raw = dataset.variables['lon'][:]
                data_raw = dataset.variables[self.material][:]
                
                # 转换为普通numpy数组
                lat = self.convert_to_array(lat_raw)
                lon = self.convert_to_array(lon_raw)
                data = self.convert_to_array(data_raw)
                
                # 验证数据维度
                if lat.shape[0] != self.expected_lat_size:
                    logger.error(f"lat维度不匹配: 期望{self.expected_lat_size}, 实际{lat.shape[0]} - {file_path}")
                    return None, None, None
                
                if lon.shape[0] != self.expected_lon_size:
                    logger.error(f"lon维度不匹配: 期望{self.expected_lon_size}, 实际{lon.shape[0]} - {file_path}")
                    return None, None, None
                
                if data.shape != (self.expected_lat_size, self.expected_lon_size):
                    logger.error(f"数据维度不匹配: 期望({self.expected_lat_size}, {self.expected_lon_size}), 实际{data.shape} - {file_path}")
                    return None, None, None
                
                return lat, lon, data
                
        except Exception as e:
            logger.error(f"加载文件失败 {file_path}: {e}")
            return None, None, None
    
    def process_year_by_year(self):
        """逐年处理数据"""
        logger.info(f"开始逐年处理{self.material}数据...")
        
        total_files_processed = 0
        total_files_expected = 0
        
        # 存储所有年份的数据
        all_years_data = []
        all_years_lat = []
        all_years_lon = []
        
        for year in self.years:
            logger.info(f"处理 {self.material} {year} 年数据...")
            
            # 查找该年的文件
            date_to_file = self.find_nc_files(year)
            
            # 生成日期列表
            date_list = self.generate_date_list(year)
            total_files_expected += len(date_list)
            
            # 为该年创建数据数组
            days_in_year = len(date_list)
            year_data = np.full((days_in_year, self.expected_lat_size, self.expected_lon_size), 
                               np.nan, dtype=np.float32)
            year_lat = np.full((days_in_year, self.expected_lat_size), np.nan, dtype=np.float32)
            year_lon = np.full((days_in_year, self.expected_lon_size), np.nan, dtype=np.float32)
            
            # 处理该年的每一天
            year_processed = 0
            for day_idx, date_str in enumerate(tqdm(date_list, desc=f"{self.material} {year}")):
                if date_str in date_to_file:
                    lat, lon, data = self.load_nc_file(date_to_file[date_str])
                    
                    if lat is not None and lon is not None and data is not None:
                        # 保存参考坐标（仅第一次）
                        if self.reference_lat is None:
                            self.reference_lat = lat.copy()
                            self.reference_lon = lon.copy()
                            logger.info(f"保存参考坐标: lat({lat.shape}), lon({lon.shape})")
                        
                        # 存储数据
                        year_lat[day_idx, :] = lat
                        year_lon[day_idx, :] = lon
                        year_data[day_idx, :, :] = data
                        
                        year_processed += 1
                        total_files_processed += 1
                    else:
                        logger.warning(f"跳过损坏的文件: {date_to_file[date_str]}")
                else:
                    logger.warning(f"缺失文件: {self.material} {date_str}")
            
            logger.info(f"{year}年处理完成: {year_processed}/{days_in_year} 文件")
            
            # 添加到总列表
            all_years_data.append(year_data)
            all_years_lat.append(year_lat)
            all_years_lon.append(year_lon)
        
        logger.info(f"所有年份处理完成: {total_files_processed}/{total_files_expected} 文件")
        return all_years_lat, all_years_lon, all_years_data
    
    def combine_years_data(self, all_years_lat, all_years_lon, all_years_data):
        """合并所有年份的数据"""
        logger.info("合并所有年份的数据...")
        
        # 找出最大天数（考虑闰年）
        max_days = max(year_data.shape[0] for year_data in all_years_data)
        n_years = len(all_years_data)
        
        logger.info(f"数据维度: {n_years}年 × 最大{max_days}天 × {self.expected_lat_size} × {self.expected_lon_size}")
        
        # 创建最终数组
        final_lat = np.full((n_years, max_days, self.expected_lat_size), np.nan, dtype=np.float32)
        final_lon = np.full((n_years, max_days, self.expected_lon_size), np.nan, dtype=np.float32)
        final_data = np.full((n_years, max_days, self.expected_lat_size, self.expected_lon_size), 
                           np.nan, dtype=np.float32)
        
        # 填充数据
        for year_idx, (year_lat, year_lon, year_data) in enumerate(zip(all_years_lat, all_years_lon, all_years_data)):
            days_in_year = year_data.shape[0]
            final_lat[year_idx, :days_in_year, :] = year_lat
            final_lon[year_idx, :days_in_year, :] = year_lon
            final_data[year_idx, :days_in_year, :, :] = year_data
        
        return final_lat, final_lon, final_data
    
    def check_coordinate_consistency(self, lat_array, lon_array):
        """检查坐标一致性"""
        logger.info("检查坐标一致性...")
        
        if self.reference_lat is None or self.reference_lon is None:
            logger.warning("缺少参考坐标")
            return False, False
        
        lat_consistent = True
        lon_consistent = True
        
        n_years, max_days = lat_array.shape[:2]
        
        # 检查前几个有效数据点的坐标
        checked_points = 0
        max_check = 100  # 只检查前100个有效点以节省时间
        
        for year_idx in range(n_years):
            for day_idx in range(max_days):
                if not np.isnan(lat_array[year_idx, day_idx, 0]) and checked_points < max_check:
                    if not np.allclose(lat_array[year_idx, day_idx, :], self.reference_lat, equal_nan=True, rtol=1e-6):
                        lat_consistent = False
                    if not np.allclose(lon_array[year_idx, day_idx, :], self.reference_lon, equal_nan=True, rtol=1e-6):
                        lon_consistent = False
                    
                    checked_points += 1
                    
                    if not lat_consistent and not lon_consistent:
                        break
            if not lat_consistent and not lon_consistent:
                break
        
        logger.info(f"坐标一致性检查（检查了{checked_points}个点）: lat={lat_consistent}, lon={lon_consistent}")
        return lat_consistent, lon_consistent
    
    def save_data(self, lat_data, lon_data, observation_data):
        """保存数据到npy文件"""
        output_dir = os.path.join(self.base_path, self.material)
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件路径
        lat_file = os.path.join(output_dir, f"CHAP_{self.material}_lat.npy")
        lon_file = os.path.join(output_dir, f"CHAP_{self.material}_lon.npy")
        obs_file = os.path.join(output_dir, f"CHAP_{self.material}_observation.npy")
        
        # 确保数据是普通numpy数组
        lat_data = self.convert_to_array(lat_data)
        lon_data = self.convert_to_array(lon_data)
        observation_data = self.convert_to_array(observation_data)
        
        logger.info(f"保存数据:")
        logger.info(f"  lat: {lat_data.shape} -> {lat_file}")
        logger.info(f"  lon: {lon_data.shape} -> {lon_file}")
        logger.info(f"  obs: {observation_data.shape} -> {obs_file}")
        
        # 保存数据
        np.save(lat_file, lat_data)
        np.save(lon_file, lon_data)
        np.save(obs_file, observation_data)
        
        # 验证保存的文件
        try:
            saved_lat = np.load(lat_file)
            saved_lon = np.load(lon_file)
            saved_obs = np.load(obs_file)
            
            logger.info(f"验证保存结果:")
            logger.info(f"  lat: {saved_lat.shape}, 文件大小: {os.path.getsize(lat_file) / (1024**2):.1f} MB")
            logger.info(f"  lon: {saved_lon.shape}, 文件大小: {os.path.getsize(lon_file) / (1024**2):.1f} MB")
            logger.info(f"  obs: {saved_obs.shape}, 文件大小: {os.path.getsize(obs_file) / (1024**2):.1f} MB")
            
            # 检查数据完整性
            obs_valid_ratio = np.sum(~np.isnan(saved_obs)) / saved_obs.size * 100
            logger.info(f"  数据完整率: {obs_valid_ratio:.2f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"验证保存文件时出错: {e}")
            return False
    
    def process_all(self):
        """执行完整的处理流程"""
        logger.info(f"开始处理{self.material}数据整合...")
        logger.info(f"基础路径: {self.base_path}")
        logger.info(f"年份范围: {self.years}")
        
        try:
            # 第一步：逐年处理数据
            all_years_lat, all_years_lon, all_years_data = self.process_year_by_year()
            
            if not all_years_data:
                logger.error("没有成功处理任何数据")
                return False
            
            # 第二步：合并所有年份数据
            combined_lat, combined_lon, combined_data = self.combine_years_data(
                all_years_lat, all_years_lon, all_years_data)
            
            # 第三步：检查坐标一致性并优化存储
            lat_consistent, lon_consistent = self.check_coordinate_consistency(combined_lat, combined_lon)
            
            if lat_consistent and lon_consistent:
                logger.info("使用优化存储格式（坐标一致）")
                success = self.save_data(self.reference_lat, self.reference_lon, combined_data)
            else:
                logger.info("使用完整存储格式（坐标变化）")
                success = self.save_data(combined_lat, combined_lon, combined_data)
            
            if success:
                logger.info(f"{self.material}数据整合完成！")
                return True
            else:
                logger.error(f"{self.material}数据保存失败！")
                return False
            
        except Exception as e:
            logger.error(f"处理过程中出错: {e}")
            return False

def main():
    """主函数"""
    # 设置基础路径
    base_path = "data/CHAP/"
    
    # 创建气体数据整合器
    material = 'SO4'
    consolidator = DataConsolidatorFull(base_path, material)
    
    # 执行整合
    success = consolidator.process_all()
    
    if success:
        print(f"{material}数据整合成功完成！")
    else:
        print(f"{material}数据整合失败！")

if __name__ == "__main__":
    main()