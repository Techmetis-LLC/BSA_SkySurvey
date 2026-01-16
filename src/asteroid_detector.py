#!/usr/bin/env python3
"""
Astronomical Object Detection System
Detects moving objects (asteroids, comets) in astronomical image sequences
by identifying objects that don't follow stellar motion patterns.
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
import astropy.visualization as vis
from astroquery.jplhorizons import Horizons
from astroquery.mpc import MPC
import sep
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture
from skimage import io as skio
from skimage.registration import phase_cross_correlation
from sklearn.cluster import DBSCAN
import cv2
from tqdm import tqdm

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='astropy.wcs')
warnings.filterwarnings('ignore', message='.*datfix.*')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles loading and preprocessing of astronomical images."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.supported_formats = ['.jpg', '.jpeg', '.tiff', '.tif', '.fits', '.fit', '.xisf']
    
    def load_image(self, filepath: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load an astronomical image and extract metadata."""
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()
        
        if suffix not in self.supported_formats:
            raise ValueError(f"Unsupported format: {suffix}")
        
        metadata = {'filepath': str(filepath), 'format': suffix}
        
        try:
            if suffix in ['.fits', '.fit']:
                return self._load_fits(filepath, metadata)
            elif suffix == '.xisf':
                return self._load_xisf(filepath, metadata)
            else:
                return self._load_standard_image(filepath, metadata)
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            raise
    
    def _load_fits(self, filepath: Path, metadata: Dict) -> Tuple[np.ndarray, Dict]:
        """Load FITS image."""
        with fits.open(filepath) as hdul:
            data = hdul[0].data.astype(np.float64)
            header = hdul[0].header
            
            # Extract WCS if present
            try:
                wcs = WCS(header)
                metadata['wcs'] = wcs
            except:
                logger.warning(f"Could not extract WCS from {filepath}")
                metadata['wcs'] = None
            
            # Extract observation time
            for time_key in ['DATE-OBS', 'DATE', 'MJD-OBS']:
                if time_key in header:
                    try:
                        metadata['obs_time'] = Time(header[time_key])
                        break
                    except:
                        continue
            
            metadata['header'] = dict(header)
            
        return data, metadata
    
    def _load_xisf(self, filepath: Path, metadata: Dict) -> Tuple[np.ndarray, Dict]:
        """Load XISF image (requires xisf library)."""
        try:
            import xisf
            xisf_file = xisf.XISF(filepath)
            data = xisf_file.read_image(0).astype(np.float64)
            metadata['xisf_metadata'] = xisf_file.get_metadata()
            return data, metadata
        except ImportError:
            raise ImportError("xisf library required for XISF support. Install with: pip install xisf")
    
    def _load_standard_image(self, filepath: Path, metadata: Dict) -> Tuple[np.ndarray, Dict]:
        """Load standard image formats (JPG, TIFF)."""
        data = skio.imread(filepath)
        
        # Convert to grayscale if needed
        if len(data.shape) == 3:
            data = np.mean(data, axis=2)
        
        return data.astype(np.float64), metadata
    
    def preprocess_image(self, data: np.ndarray, metadata: Dict) -> np.ndarray:
        """Apply preprocessing to image data."""
        # Basic preprocessing
        processed = data.copy()
        
        # Remove hot pixels and cosmic rays
        processed = self._remove_cosmic_rays(processed)
        
        # Background subtraction
        processed = self._subtract_background(processed)
        
        if self.debug:
            logger.debug(f"Image preprocessed: shape={processed.shape}, "
                        f"min={processed.min():.2f}, max={processed.max():.2f}")
        
        return processed
    
    def _remove_cosmic_rays(self, data: np.ndarray) -> np.ndarray:
        """Simple cosmic ray removal using median filtering."""
        from scipy import ndimage
        median_filtered = ndimage.median_filter(data, size=3)
        diff = np.abs(data - median_filtered)
        threshold = 5 * np.std(diff)
        cosmic_rays = diff > threshold
        
        result = data.copy()
        result[cosmic_rays] = median_filtered[cosmic_rays]
        
        return result
    
    def _subtract_background(self, data: np.ndarray) -> np.ndarray:
        """Subtract background using SEP - NumPy 2.0 compatible."""
        try:
            # Ensure data is in correct byte order and type
            data_copy = np.ascontiguousarray(data, dtype=np.float64)
            
            # For NumPy 2.0+ compatibility, ensure native byte order
            if data_copy.dtype.byteorder not in ('=', '|'):
                # Convert to native byte order
                data_copy = data_copy.astype(data_copy.dtype.newbyteorder('='))
            
            bkg = sep.Background(data_copy)
            return data - bkg.back()
        except Exception as e:
            logger.warning(f"SEP background subtraction failed: {e}, using simple method")
            # Fallback to simple background subtraction
            from scipy import ndimage
            background = ndimage.gaussian_filter(data, sigma=50)
            return data - background


class StarDetector:
    """Detects and measures stars in astronomical images."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def detect_stars(self, data: np.ndarray, threshold: float = 5.0) -> Table:
        """Detect stars in the image."""
        try:
            # Ensure data is in correct format for SEP
            data_copy = np.ascontiguousarray(data, dtype=np.float64)
            
            # For NumPy 2.0+ compatibility
            if data_copy.dtype.byteorder not in ('=', '|'):
                data_copy = data_copy.astype(data_copy.dtype.newbyteorder('='))
            
            # Use SEP for source detection
            bkg = sep.Background(data_copy)
            bkg_subtracted = data_copy - bkg.back()
            
            objects = sep.extract(bkg_subtracted, threshold * bkg.globalrms)
            
            # Convert to astropy Table
            sources = Table()
            sources['x'] = objects['x']
            sources['y'] = objects['y']
            sources['flux'] = objects['flux']
            sources['a'] = objects['a']  # semi-major axis
            sources['b'] = objects['b']  # semi-minor axis
            sources['theta'] = objects['theta']
            sources['flag'] = objects['flag']
            
            # Calculate signal-to-noise ratio
            sources['snr'] = sources['flux'] / np.sqrt(sources['flux'] + bkg.globalrms**2)
            
            # Filter out likely non-stellar objects
            sources = self._filter_stellar_objects(sources, data.shape)
            
            if self.debug:
                logger.debug(f"Detected {len(sources)} stellar objects")
            
            return sources
            
        except Exception as e:
            logger.error(f"Star detection failed: {e}")
            # Fallback to DAOStarFinder
            return self._fallback_star_detection(data, threshold)
    
    def _fallback_star_detection(self, data: np.ndarray, threshold: float) -> Table:
        """Fallback star detection using photutils."""
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        daofind = DAOStarFinder(fwhm=3.0, threshold=threshold * std)
        sources = daofind(data - median)
        
        if sources is None:
            return Table()
        
        sources['snr'] = sources['peak'] / std
        return sources
    
    def _filter_stellar_objects(self, sources: Table, image_shape: Tuple) -> Table:
        """Filter to keep only stellar-like objects."""
        if len(sources) == 0:
            return sources
        
        # Remove flagged objects
        mask = sources['flag'] == 0
        
        # Remove objects that are too elongated (likely galaxies or artifacts)
        ellipticity = 1 - sources['b'] / (sources['a'] + 1e-10)  # Avoid division by zero
        mask &= ellipticity < 0.5
        
        # Remove very faint objects
        mask &= sources['snr'] > 10
        
        # Remove objects near image edges
        height, width = image_shape
        edge_buffer = 50
        mask &= (sources['x'] > edge_buffer) & (sources['x'] < width - edge_buffer)
        mask &= (sources['y'] > edge_buffer) & (sources['y'] < height - edge_buffer)
        
        return sources[mask]


class MotionDetector:
    """Detects objects with motion different from stellar motion."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.reference_stars = None
        self.transformation_matrix = None
    
    def register_images(self, image_data: List[Tuple[np.ndarray, Table, Dict]]) -> List[np.ndarray]:
        """Register images to a common reference frame."""
        if len(image_data) < 2:
            raise ValueError("Need at least 2 images for motion detection")
        
        reference_data, reference_sources, _ = image_data[0]
        registered_images = [reference_data]
        
        # Use brightest stars as reference points
        if len(reference_sources) > 0:
            n_stars = min(50, len(reference_sources))
            ref_stars = reference_sources[np.argsort(reference_sources['flux'])[-n_stars:]]
            self.reference_stars = np.column_stack([ref_stars['x'], ref_stars['y']])
        else:
            logger.warning("No reference stars found in first image")
            self.reference_stars = np.array([])
        
        for i, (data, sources, metadata) in enumerate(image_data[1:], 1):
            if self.debug:
                logger.debug(f"Registering image {i+1}/{len(image_data)}")
            
            # Find corresponding stars
            if len(sources) > 0 and len(self.reference_stars) > 0:
                n_stars = min(50, len(sources))
                curr_stars = sources[np.argsort(sources['flux'])[-n_stars:]]
                curr_positions = np.column_stack([curr_stars['x'], curr_stars['y']])
                
                # Calculate transformation
                transform = self._calculate_transformation(self.reference_stars, curr_positions)
                
                # Apply transformation
                registered = self._apply_transformation(data, transform)
                registered_images.append(registered)
            else:
                logger.warning(f"Insufficient stars for registration in image {i+1}")
                registered_images.append(data)
        
        return registered_images
    
    def _calculate_transformation(self, ref_points: np.ndarray, curr_points: np.ndarray) -> np.ndarray:
        """Calculate transformation matrix between point sets."""
        # Simple translation-based registration
        from scipy.spatial.distance import cdist
        
        distances = cdist(ref_points, curr_points)
        matches = []
        
        for i in range(min(len(ref_points), len(curr_points), 10)):
            ref_idx, curr_idx = np.unravel_index(distances.argmin(), distances.shape)
            matches.append((ref_points[ref_idx], curr_points[curr_idx]))
            distances[ref_idx, :] = np.inf
            distances[:, curr_idx] = np.inf
        
        if len(matches) < 3:
            return np.eye(3)  # Identity matrix if not enough matches
        
        # Calculate average translation
        ref_matched = np.array([m[0] for m in matches])
        curr_matched = np.array([m[1] for m in matches])
        
        translation = np.mean(ref_matched - curr_matched, axis=0)
        
        # Create transformation matrix
        transform = np.eye(3)
        transform[0:2, 2] = translation
        
        return transform
    
    def _apply_transformation(self, image: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Apply transformation to image."""
        # Extract translation for simple case
        translation = transform[0:2, 2]
        
        # Use OpenCV for image translation
        M = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
        registered = cv2.warpAffine(image.astype(np.float32), M, 
                                   (image.shape[1], image.shape[0]), 
                                   flags=cv2.INTER_LINEAR)
        
        return registered.astype(np.float64)
    
    def detect_moving_objects(self, registered_images: List[np.ndarray], 
                            source_tables: List[Table]) -> List[Dict]:
        """Detect objects that move between images."""
        moving_objects = []
        
        if len(registered_images) < 2:
            return moving_objects
        
        # Create difference images
        diff_images = []
        for i in range(1, len(registered_images)):
            diff = registered_images[i] - registered_images[0]
            diff_images.append(diff)
        
        # Detect sources in difference images
        for i, diff_img in enumerate(diff_images):
            # Smooth difference image to reduce noise
            from scipy import ndimage
            smoothed_diff = ndimage.gaussian_filter(np.abs(diff_img), sigma=1.0)
            
            # Find peaks in difference image
            threshold = 3 * np.std(smoothed_diff)
            peaks = self._find_peaks(smoothed_diff, threshold)
            
            for peak in peaks:
                x, y = peak
                
                # Verify this isn't a known star
                if not self._is_known_star(x, y, source_tables[0]):
                    obj = {
                        'x': x,
                        'y': y,
                        'image_pair': (0, i + 1),
                        'flux_diff': diff_img[int(y), int(x)],
                        'detection_time': i
                    }
                    moving_objects.append(obj)
        
        # Cluster detections that might be the same object
        if moving_objects:
            moving_objects = self._cluster_detections(moving_objects)
        
        if self.debug:
            logger.debug(f"Found {len(moving_objects)} potential moving objects")
        
        return moving_objects
    
    def _find_peaks(self, image: np.ndarray, threshold: float) -> List[Tuple[float, float]]:
        """Find peaks in image above threshold."""
        from scipy.ndimage import maximum_filter
        from scipy.ndimage import binary_erosion
        
        # Find local maxima
        local_maxima = maximum_filter(image, size=5) == image
        background = image < threshold
        eroded_background = binary_erosion(background, structure=np.ones((3, 3)))
        detected_peaks = local_maxima ^ eroded_background
        
        # Get peak coordinates
        y_coords, x_coords = np.where(detected_peaks)
        peaks = list(zip(x_coords.astype(float), y_coords.astype(float)))
        
        return peaks
    
    def _is_known_star(self, x: float, y: float, star_catalog: Table, tolerance: float = 5.0) -> bool:
        """Check if position corresponds to a known star."""
        if len(star_catalog) == 0:
            return False
        
        distances = np.sqrt((star_catalog['x'] - x)**2 + (star_catalog['y'] - y)**2)
        return np.any(distances < tolerance)
    
    def _cluster_detections(self, detections: List[Dict]) -> List[Dict]:
        """Cluster detections that likely belong to the same object."""
        if len(detections) < 2:
            return detections
        
        # Prepare data for clustering
        positions = np.array([[det['x'], det['y']] for det in detections])
        
        # Use DBSCAN to cluster nearby detections
        clustering = DBSCAN(eps=10, min_samples=1).fit(positions)
        labels = clustering.labels_
        
        # Group detections by cluster
        clustered_objects = []
        for label in set(labels):
            cluster_detections = [det for i, det in enumerate(detections) if labels[i] == label]
            
            # Calculate average position for cluster
            avg_x = np.mean([det['x'] for det in cluster_detections])
            avg_y = np.mean([det['y'] for det in cluster_detections])
            
            clustered_obj = {
                'x': avg_x,
                'y': avg_y,
                'detections': cluster_detections,
                'n_detections': len(cluster_detections)
            }
            clustered_objects.append(clustered_obj)
        
        return clustered_objects


class PlateSolver:
    """Handles plate solving for coordinate determination."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def solve_field(self, image_data: np.ndarray, metadata: Dict) -> Optional[WCS]:
        """Solve the astrometric solution for an image."""
        # If WCS already exists in metadata, use it
        if 'wcs' in metadata and metadata['wcs'] is not None:
            if self.debug:
                logger.debug("Using existing WCS solution")
            return metadata['wcs']
        
        # Try to use astrometry.net for plate solving
        try:
            return self._solve_with_astrometry_net(image_data, metadata)
        except Exception as e:
            logger.warning(f"Plate solving failed: {e}")
            return None
    
    def _solve_with_astrometry_net(self, image_data: np.ndarray, metadata: Dict) -> Optional[WCS]:
        """Solve using astrometry.net (requires astroquery and API key)."""
        # This is a placeholder - actual implementation would require
        # astrometry.net API integration or local installation
        logger.warning("Astrometry.net integration not implemented in this example")
        return None
    
    def pixel_to_world(self, wcs: WCS, x: float, y: float) -> SkyCoord:
        """Convert pixel coordinates to world coordinates."""
        if wcs is None:
            raise ValueError("No WCS solution available")
        
        world_coords = wcs.pixel_to_world(x, y)
        return world_coords


class ObjectIdentifier:
    """Identifies objects using astronomical databases."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def query_minor_planet_center(self, coord: SkyCoord, radius: float = 0.1) -> List[Dict]:
        """Query Minor Planet Center for known objects."""
        try:
            # Query MPC for objects near the coordinates
            result = MPC.query_region(coord, radius=radius * u.deg)
            
            objects = []
            if result is not None:
                for row in result:
                    obj = {
                        'name': row.get('name', 'Unknown'),
                        'designation': row.get('designation', ''),
                        'object_type': 'asteroid',
                        'database': 'MPC',
                        'separation': 0.0  # Would calculate actual separation
                    }
                    objects.append(obj)
            
            return objects
            
        except Exception as e:
            logger.error(f"MPC query failed: {e}")
            return []
    
    def query_jpl_horizons(self, coord: SkyCoord, obs_time: Time, radius: float = 0.1) -> List[Dict]:
        """Query JPL Horizons for known objects."""
        try:
            # This would require more complex implementation
            # to search for objects near given coordinates
            logger.warning("JPL Horizons coordinate search not fully implemented")
            return []
            
        except Exception as e:
            logger.error(f"JPL Horizons query failed: {e}")
            return []
    
    def identify_object(self, coord: SkyCoord, obs_time: Optional[Time] = None) -> Dict:
        """Attempt to identify an object at given coordinates."""
        identification = {
            'coordinates': coord,
            'known_objects': [],
            'is_known': False,
            'best_match': None
        }
        
        # Query databases
        mpc_results = self.query_minor_planet_center(coord)
        identification['known_objects'].extend(mpc_results)
        
        if obs_time:
            horizons_results = self.query_jpl_horizons(coord, obs_time)
            identification['known_objects'].extend(horizons_results)
        
        if identification['known_objects']:
            identification['is_known'] = True
            identification['best_match'] = identification['known_objects'][0]
        
        return identification


class AsteroidDetector:
    """Main class coordinating the detection pipeline."""
    
    def __init__(self, debug: bool = False, progress: bool = True):
        self.debug = debug
        self.show_progress = progress
        
        self.image_processor = ImageProcessor(debug)
        self.star_detector = StarDetector(debug)
        self.motion_detector = MotionDetector(debug)
        self.plate_solver = PlateSolver(debug)
        self.object_identifier = ObjectIdentifier(debug)
        
        self.results = []
    
    def process_image_sequence(self, image_paths: List[Path]) -> Dict:
        """Process a sequence of images to detect moving objects."""
        if len(image_paths) < 2:
            raise ValueError("Need at least 2 images for motion detection")
        
        logger.info(f"Processing {len(image_paths)} images")
        
        # Load and preprocess images
        image_data = []
        
        with tqdm(total=len(image_paths), desc="Loading images", 
                 disable=not self.show_progress) as pbar:
            for path in image_paths:
                try:
                    data, metadata = self.image_processor.load_image(path)
                    processed_data = self.image_processor.preprocess_image(data, metadata)
                    
                    # Detect stars
                    sources = self.star_detector.detect_stars(processed_data)
                    
                    image_data.append((processed_data, sources, metadata))
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Failed to process {path}: {e}")
                    if self.debug:
                        raise
                    continue
        
        if len(image_data) < 2:
            raise RuntimeError("Not enough valid images processed")
        
        # Register images
        logger.info("Registering images")
        registered_images = self.motion_detector.register_images(image_data)
        
        # Detect moving objects
        logger.info("Detecting moving objects")
        source_tables = [item[1] for item in image_data]
        moving_objects = self.motion_detector.detect_moving_objects(
            registered_images, source_tables)
        
        # Plate solve and identify objects
        results = []
        wcs = None
        
        if moving_objects:
            logger.info(f"Analyzing {len(moving_objects)} potential objects")
            
            # Get WCS solution from first image
            wcs = self.plate_solver.solve_field(image_data[0][0], image_data[0][2])
            
            with tqdm(total=len(moving_objects), desc="Identifying objects",
                     disable=not self.show_progress) as pbar:
                for obj in moving_objects:
                    result = self._analyze_object(obj, wcs, image_data[0][2])
                    results.append(result)
                    pbar.update(1)
        
        # Compile final results
        summary = {
            'n_images_processed': len(image_data),
            'n_moving_objects_detected': len(moving_objects),
            'n_identified_objects': sum(1 for r in results if r['identification']['is_known']),
            'n_unknown_objects': sum(1 for r in results if not r['identification']['is_known']),
            'wcs_available': wcs is not None,
            'objects': results
        }
        
        self.results = summary
        return summary
    
    def _analyze_object(self, obj: Dict, wcs: Optional[WCS], metadata: Dict) -> Dict:
        """Analyze a detected moving object."""
        result = {
            'pixel_coordinates': (obj['x'], obj['y']),
            'world_coordinates': None,
            'identification': None,
            'confidence': 'low',
            'metadata': obj
        }
        
        # Convert to world coordinates if possible
        if wcs is not None:
            try:
                world_coord = self.plate_solver.pixel_to_world(wcs, obj['x'], obj['y'])
                result['world_coordinates'] = world_coord
                
                # Attempt identification
                obs_time = metadata.get('obs_time')
                identification = self.object_identifier.identify_object(world_coord, obs_time)
                result['identification'] = identification
                
                # Set confidence based on number of detections and identification
                if obj.get('n_detections', 1) > 2:
                    result['confidence'] = 'high' if identification['is_known'] else 'medium'
                
            except Exception as e:
                logger.error(f"Failed to analyze object: {e}")
                if self.debug:
                    raise
        
        return result
    
    def save_results(self, output_path: Path):
        """Save results to JSON file."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        # Convert results to JSON-serializable format
        serializable_results = self._make_json_serializable(self.results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif hasattr(obj, 'to_string'):  # SkyCoord objects
            return str(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def generate_report(self, output_path: Path):
        """Generate a detailed analysis report."""
        if not self.results:
            logger.warning("No results available for report")
            return
        
        report_lines = [
            "# Astronomical Object Detection Report",
            f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- Images processed: {self.results['n_images_processed']}",
            f"- Moving objects detected: {self.results['n_moving_objects_detected']}",
            f"- Known objects identified: {self.results['n_identified_objects']}",
            f"- Unknown objects: {self.results['n_unknown_objects']}",
            f"- Plate solving successful: {self.results['wcs_available']}",
            "",
            "## Detected Objects",
        ]
        
        for i, obj in enumerate(self.results['objects'], 1):
            report_lines.extend([
                f"### Object {i}",
                f"- Pixel coordinates: ({obj['pixel_coordinates'][0]:.1f}, {obj['pixel_coordinates'][1]:.1f})",
                f"- Confidence: {obj['confidence']}",
            ])
            
            if obj['world_coordinates']:
                report_lines.append(f"- Sky coordinates: {obj['world_coordinates']}")
            
            if obj['identification']:
                id_info = obj['identification']
                if id_info['is_known']:
                    best_match = id_info['best_match']
                    report_lines.extend([
                        f"- **IDENTIFIED**: {best_match['name']}",
                        f"- Object type: {best_match['object_type']}",
                        f"- Database: {best_match['database']}",
                    ])
                else:
                    report_lines.append("- **UNKNOWN OBJECT** - Potential new discovery!")
            
            report_lines.append("")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Report saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect moving objects in astronomical image sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s images/*.fits
  %(prog)s --debug --output results.json image1.jpg image2.jpg image3.jpg
  %(prog)s --no-progress --threshold 3.0 *.tiff
        """
    )
    
    parser.add_argument('images', nargs='+', help='Input image files')
    parser.add_argument('--output', '-o', type=Path, default='detection_results.json',
                       help='Output JSON file (default: detection_results.json)')
    parser.add_argument('--report', '-r', type=Path, default='detection_report.md',
                       help='Output report file (default: detection_report.md)')
    parser.add_argument('--threshold', '-t', type=float, default=5.0,
                       help='Detection threshold (default: 5.0)')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bars')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert image paths
    image_paths = [Path(p) for p in args.images]
    
    # Validate input files
    for path in image_paths:
        if not path.exists():
            logger.error(f"File not found: {path}")
            sys.exit(1)
    
    try:
        # Create detector
        detector = AsteroidDetector(debug=args.debug, progress=not args.no_progress)
        
        # Process images
        results = detector.process_image_sequence(image_paths)
        
        # Save results
        detector.save_results(args.output)
        detector.generate_report(args.report)
        
        # Print summary
        print(f"\n=== Detection Summary ===")
        print(f"Images processed: {results['n_images_processed']}")
        print(f"Moving objects detected: {results['n_moving_objects_detected']}")
        print(f"Known objects identified: {results['n_identified_objects']}")
        print(f"Unknown objects: {results['n_unknown_objects']}")
        
        if results['n_unknown_objects'] > 0:
            print(f"\n🎉 Found {results['n_unknown_objects']} potential new discoveries!")
        
        print(f"\nResults saved to: {args.output}")
        print(f"Report saved to: {args.report}")
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        if args.debug:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()