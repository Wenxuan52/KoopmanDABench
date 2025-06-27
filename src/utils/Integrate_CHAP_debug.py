import os
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta
import glob
from tqdm import tqdm
import logging

# 设置详细日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClDataConsolidatorDebug:
    def __init__(self, base_path):
        """
        Cl气体数据整合器 - 调试版本
        """
        self.base_path = base_path
        self.material = 'Cl'
        self.years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        
        # 预期的数据维度
        self.expected_lat_size = 2893
        self.expected_lon_size = 2439
        
        # 用于存储参考坐标
        self.reference_lat = None
        self.reference_lon = None
        
        logger.info(f"初始化调试器 - 基础路径: {self.base_path}")
    
    def debug_single_file(self):
        """调试单个文件加载"""
        logger.info("=== 调试单个文件加载 ===")
        
        # 查找第一个可用文件
        test_file = None
        for year in [2013]:  # 只测试2013年
            year_path = os.path.join(self.base_path, self.material, str(year))
            if os.path.exists(year_path):
                pattern = os.path.join(year_path, f"ECHAP_{self.material}_D1K_{year}0101_V1.nc")
                files = glob.glob(pattern)
                if files:
                    test_file = files[0]
                    break
        
        if not test_file:
            logger.error("未找到测试文件")
            return False
        
        logger.info(f"测试文件: {test_file}")
        
        try:
            with nc.Dataset(test_file, 'r') as dataset:
                logger.info(f"文件变量: {list(dataset.variables.keys())}")
                logger.info(f"文件维度: {dict(dataset.dimensions)}")
                
                for var_name in dataset.variables:
                    var = dataset.variables[var_name]
                    logger.info(f"变量 {var_name}: 形状={var.shape}, 类型={var.dtype}")
                
                # 尝试读取数据
                if 'lat' in dataset.variables:
                    lat_raw = dataset.variables['lat']
                    logger.info(f"lat变量类型: {type(lat_raw)}")
                    lat_data = lat_raw[:]
                    logger.info(f"lat数据类型: {type(lat_data)}")
                    logger.info(f"lat形状: {lat_data.shape}")
                    logger.info(f"lat是否为MaskedArray: {hasattr(lat_data, 'filled')}")
                    
                    # 处理MaskedArray
                    if hasattr(lat_data, 'filled'):
                        logger.info("检测到MaskedArray，转换为普通数组")
                        lat_converted = lat_data.filled(np.nan)
                        logger.info(f"转换后类型: {type(lat_converted)}")
                    else:
                        lat_converted = lat_data
                    
                    # 确保为numpy数组
                    lat_final = np.asarray(lat_converted, dtype=np.float32)
                    logger.info(f"最终数组类型: {type(lat_final)}")
                    logger.info(f"最终数组形状: {lat_final.shape}")
                    logger.info(f"lat范围: {np.nanmin(lat_final)} - {np.nanmax(lat_final)}")
                
                if 'lon' in dataset.variables:
                    lon_raw = dataset.variables['lon']
                    lon_data = lon_raw[:]
                    logger.info(f"lon是否为MaskedArray: {hasattr(lon_data, 'filled')}")
                    
                    if hasattr(lon_data, 'filled'):
                        lon_converted = lon_data.filled(np.nan)
                    else:
                        lon_converted = lon_data
                    
                    lon_final = np.asarray(lon_converted, dtype=np.float32)
                    logger.info(f"lon范围: {np.nanmin(lon_final)} - {np.nanmax(lon_final)}")
                
                if self.material in dataset.variables:
                    data_raw = dataset.variables[self.material]
                    data_array = data_raw[:]
                    logger.info(f"{self.material}是否为MaskedArray: {hasattr(data_array, 'filled')}")
                    
                    if hasattr(data_array, 'filled'):
                        data_converted = data_array.filled(np.nan)
                    else:
                        data_converted = data_array
                    
                    data_final = np.asarray(data_converted, dtype=np.float32)
                    logger.info(f"{self.material}范围: {np.nanmin(data_final)} - {np.nanmax(data_final)}")
                    logger.info(f"{self.material}缺失值数量: {np.sum(np.isnan(data_final))}")
                
                return True
                
        except Exception as e:
            logger.error(f"调试单个文件时出错: {e}")
            return False
    
    def test_small_array_save(self):
        """测试小数组保存"""
        logger.info("=== 测试小数组保存 ===")
        
        try:
            # 创建测试数组
            test_lat = np.random.rand(100).astype(np.float32)
            test_lon = np.random.rand(100).astype(np.float32)
            test_data = np.random.rand(10, 100, 100).astype(np.float32)
            
            # 确保输出目录存在
            output_dir = os.path.join(self.base_path, self.material)
            os.makedirs(output_dir, exist_ok=True)
            
            # 测试保存
            test_files = {
                'lat': os.path.join(output_dir, "test_lat.npy"),
                'lon': os.path.join(output_dir, "test_lon.npy"),
                'data': os.path.join(output_dir, "test_data.npy")
            }
            
            logger.info(f"测试保存到目录: {output_dir}")
            
            np.save(test_files['lat'], test_lat)
            np.save(test_files['lon'], test_lon)
            np.save(test_files['data'], test_data)
            
            # 验证保存
            for name, file_path in test_files.items():
                if os.path.exists(file_path):
                    loaded = np.load(file_path)
                    logger.info(f"测试{name}保存成功: 形状={loaded.shape}, 大小={os.path.getsize(file_path)}字节")
                else:
                    logger.error(f"测试{name}保存失败")
                    return False
            
            # 清理测试文件
            for file_path in test_files.values():
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            return True
            
        except Exception as e:
            logger.error(f"测试保存时出错: {e}")
            return False
    
    def process_single_year_debug(self, year=2013, max_files=5):
        """调试处理单年数据（限制文件数量）"""
        logger.info(f"=== 调试处理{year}年数据（最多{max_files}个文件）===")
        
        # 查找文件
        year_path = os.path.join(self.base_path, self.material, str(year))
        if not os.path.exists(year_path):
            logger.error(f"年份目录不存在: {year_path}")
            return None
        
        pattern = os.path.join(year_path, f"ECHAP_{self.material}_D1K_{year}*_V1.nc")
        all_files = glob.glob(pattern)
        logger.info(f"找到{len(all_files)}个文件")
        
        # 限制文件数量用于调试
        test_files = all_files[:max_files]
        logger.info(f"调试处理前{len(test_files)}个文件")
        
        # 创建数据容器
        processed_data = []
        processed_lat = []
        processed_lon = []
        
        for i, file_path in enumerate(test_files):
            logger.info(f"处理文件 {i+1}/{len(test_files)}: {os.path.basename(file_path)}")
            
            try:
                with nc.Dataset(file_path, 'r') as dataset:
                    # 读取并处理数据
                    lat_raw = dataset.variables['lat'][:]
                    lon_raw = dataset.variables['lon'][:]
                    data_raw = dataset.variables[self.material][:]
                    
                    # 处理MaskedArray
                    lat = self._convert_to_array(lat_raw)
                    lon = self._convert_to_array(lon_raw)
                    data = self._convert_to_array(data_raw)
                    
                    # 验证数据
                    if self._validate_data(lat, lon, data, file_path):
                        processed_lat.append(lat)
                        processed_lon.append(lon)
                        processed_data.append(data)
                        
                        # 保存参考坐标
                        if self.reference_lat is None:
                            self.reference_lat = lat.copy()
                            self.reference_lon = lon.copy()
                            logger.info("保存参考坐标")
                    else:
                        logger.warning(f"数据验证失败，跳过文件: {file_path}")
                        
            except Exception as e:
                logger.error(f"处理文件失败 {file_path}: {e}")
                continue
        
        if processed_data:
            logger.info(f"成功处理{len(processed_data)}个文件")
            
            # 转换为numpy数组
            stacked_lat = np.stack(processed_lat, axis=0)
            stacked_lon = np.stack(processed_lon, axis=0)
            stacked_data = np.stack(processed_data, axis=0)
            
            logger.info(f"堆叠数组形状:")
            logger.info(f"  lat: {stacked_lat.shape}")
            logger.info(f"  lon: {stacked_lon.shape}")
            logger.info(f"  data: {stacked_data.shape}")
            
            return stacked_lat, stacked_lon, stacked_data
        else:
            logger.error("没有成功处理任何文件")
            return None
    
    def _convert_to_array(self, data):
        """安全地将数据转换为numpy数组"""
        try:
            # 处理MaskedArray
            if hasattr(data, 'filled'):
                converted = data.filled(np.nan)
            else:
                converted = data
            
            # 转换为numpy数组
            result = np.asarray(converted, dtype=np.float32)
            return result
            
        except Exception as e:
            logger.error(f"数组转换失败: {e}")
            raise
    
    def _validate_data(self, lat, lon, data, file_path):
        """验证数据的有效性"""
        try:
            # 检查形状
            if lat.shape[0] != self.expected_lat_size:
                logger.error(f"lat维度错误: {lat.shape[0]} vs {self.expected_lat_size}")
                return False
            
            if lon.shape[0] != self.expected_lon_size:
                logger.error(f"lon维度错误: {lon.shape[0]} vs {self.expected_lon_size}")
                return False
            
            if data.shape != (self.expected_lat_size, self.expected_lon_size):
                logger.error(f"data维度错误: {data.shape} vs ({self.expected_lat_size}, {self.expected_lon_size})")
                return False
            
            # 检查数据类型
            if not np.issubdtype(lat.dtype, np.floating):
                logger.error(f"lat数据类型错误: {lat.dtype}")
                return False
            
            # 检查数据范围
            if np.all(np.isnan(data)):
                logger.warning(f"数据全为NaN: {file_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"数据验证失败: {e}")
            return False
    
    def debug_save_data(self, lat_data, lon_data, observation_data):
        """调试版本的数据保存"""
        logger.info("=== 调试保存数据 ===")
        
        try:
            # 确保输出目录存在
            output_dir = os.path.join(self.base_path, self.material)
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"输出目录: {output_dir}")
            
            # 检查数据类型
            logger.info(f"保存前数据检查:")
            logger.info(f"  lat类型: {type(lat_data)}, 形状: {lat_data.shape}, dtype: {lat_data.dtype}")
            logger.info(f"  lon类型: {type(lon_data)}, 形状: {lon_data.shape}, dtype: {lon_data.dtype}")
            logger.info(f"  obs类型: {type(observation_data)}, 形状: {observation_data.shape}, dtype: {observation_data.dtype}")
            
            # 再次确保是普通numpy数组
            lat_safe = self._convert_to_array(lat_data)
            lon_safe = self._convert_to_array(lon_data)
            obs_safe = self._convert_to_array(observation_data)
            
            logger.info(f"转换后数据检查:")
            logger.info(f"  lat类型: {type(lat_safe)}, 形状: {lat_safe.shape}")
            logger.info(f"  lon类型: {type(lon_safe)}, 形状: {lon_safe.shape}")
            logger.info(f"  obs类型: {type(obs_safe)}, 形状: {obs_safe.shape}")
            
            # 生成文件路径
            files = {
                'lat': os.path.join(output_dir, f"CHAP_{self.material}_lat_debug.npy"),
                'lon': os.path.join(output_dir, f"CHAP_{self.material}_lon_debug.npy"),
                'obs': os.path.join(output_dir, f"CHAP_{self.material}_observation_debug.npy")
            }
            
            # 逐个保存并验证
            for name, data, file_path in [
                ('lat', lat_safe, files['lat']),
                ('lon', lon_safe, files['lon']),
                ('obs', obs_safe, files['obs'])
            ]:
                logger.info(f"保存{name}数据到: {file_path}")
                
                # 保存
                np.save(file_path, data)
                
                # 立即验证
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    logger.info(f"  {name}文件已创建，大小: {file_size}字节 ({file_size/(1024**2):.2f}MB)")
                    
                    # 尝试加载验证
                    try:
                        loaded = np.load(file_path)
                        logger.info(f"  {name}验证成功: 形状={loaded.shape}")
                    except Exception as e:
                        logger.error(f"  {name}验证失败: {e}")
                        return False
                else:
                    logger.error(f"  {name}文件创建失败")
                    return False
            
            logger.info("所有数据保存并验证成功！")
            return True
            
        except Exception as e:
            logger.error(f"保存数据时出错: {e}")
            return False
    
    def run_debug_workflow(self):
        """运行完整的调试工作流"""
        logger.info("开始调试工作流...")
        
        # 步骤1：调试单个文件
        if not self.debug_single_file():
            logger.error("单文件调试失败")
            return False
        
        # 步骤2：测试小数组保存
        if not self.test_small_array_save():
            logger.error("小数组保存测试失败")
            return False
        
        # 步骤3：处理少量真实数据
        result = self.process_single_year_debug(year=2013, max_files=3)
        if result is None:
            logger.error("真实数据处理失败")
            return False
        
        lat_data, lon_data, obs_data = result
        
        # 步骤4：保存真实数据
        if not self.debug_save_data(lat_data, lon_data, obs_data):
            logger.error("真实数据保存失败")
            return False
        
        logger.info("调试工作流完成！")
        return True

def main():
    """主函数"""
    # 设置基础路径
    base_path = "data/CHAP/"
    
    # 创建调试器
    debug_consolidator = ClDataConsolidatorDebug(base_path)
    
    # 运行调试
    success = debug_consolidator.run_debug_workflow()
    
    if success:
        print("调试成功完成！")
    else:
        print("调试失败！")

if __name__ == "__main__":
    main()