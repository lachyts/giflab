#!/usr/bin/env python3
"""Standalone configuration validation tool for GifLab.

This script validates configuration files and profiles, checking for:
- Type correctness
- Range validation
- Resource availability
- Configuration relationships
- Profile compatibility
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.giflab.config_validator import (
    ConfigurationValidator,
    validate_config_file,
)
from src.giflab.config_manager import (
    ConfigManager,
    ConfigProfile,
    get_config_manager,
)


def print_validation_results(results: Dict[str, List[str]], verbose: bool = False):
    """Print validation results with formatting."""
    
    # Color codes for terminal output
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    
    has_errors = bool(results.get("error"))
    has_warnings = bool(results.get("warning"))
    has_info = bool(results.get("info"))
    
    if not has_errors and not has_warnings and not (verbose and has_info):
        print(f"{GREEN}✓ Configuration is valid{RESET}")
        return 0
    
    # Print errors
    if has_errors:
        print(f"\n{RED}Errors ({len(results['error'])}):{RESET}")
        for error in results["error"]:
            print(f"  {RED}✗{RESET} {error}")
    
    # Print warnings
    if has_warnings:
        print(f"\n{YELLOW}Warnings ({len(results['warning'])}):{RESET}")
        for warning in results["warning"]:
            print(f"  {YELLOW}⚠{RESET} {warning}")
    
    # Print info (only in verbose mode)
    if verbose and has_info:
        print(f"\n{BLUE}Info ({len(results['info'])}):{RESET}")
        for info in results["info"]:
            print(f"  {BLUE}ℹ{RESET} {info}")
    
    return 1 if has_errors else 0


def validate_file(file_path: Path, verbose: bool = False) -> int:
    """Validate a configuration file."""
    print(f"Validating configuration file: {file_path}")
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return 1
    
    try:
        results = validate_config_file(file_path)
        return print_validation_results(results, verbose)
    except Exception as e:
        print(f"Error validating file: {e}")
        return 1


def validate_profile(profile_name: str, verbose: bool = False) -> int:
    """Validate a configuration profile."""
    print(f"Validating profile: {profile_name}")
    
    try:
        # Load the profile
        profile = ConfigProfile(profile_name)
        manager = get_config_manager()
        
        # Try to load the profile
        manager.load_profile(profile)
        
        # Get the configuration and validate
        config = manager.config
        validator = ConfigurationValidator()
        
        # Validate
        results = validator.validate(config)
        
        # Check relationships
        relationship_errors = validator.validate_relationships(config)
        if relationship_errors:
            if "error" not in results:
                results["error"] = []
            results["error"].extend(relationship_errors)
        
        # Check resources
        resource_warnings = validator.check_resources(config)
        if resource_warnings:
            if "warning" not in results:
                results["warning"] = []
            results["warning"].extend(resource_warnings)
        
        return print_validation_results(results, verbose)
        
    except Exception as e:
        print(f"Error validating profile: {e}")
        return 1


def validate_current(verbose: bool = False) -> int:
    """Validate the current active configuration."""
    print("Validating current configuration...")
    
    try:
        manager = get_config_manager()
        config = manager.config
        metadata = manager.get_metadata()
        
        # Print metadata
        print(f"  Version: {metadata.version}")
        if metadata.active_profile:
            print(f"  Active Profile: {metadata.active_profile.value}")
        if metadata.last_reload:
            from datetime import datetime
            reload_time = datetime.fromtimestamp(metadata.last_reload)
            print(f"  Last Reload: {reload_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Reload Count: {metadata.reload_count}")
        if metadata.checksum:
            print(f"  Checksum: {metadata.checksum}")
        
        # Validate
        validator = ConfigurationValidator()
        results = validator.validate(config)
        
        # Check relationships
        relationship_errors = validator.validate_relationships(config)
        if relationship_errors:
            if "error" not in results:
                results["error"] = []
            results["error"].extend(relationship_errors)
        
        # Check resources
        resource_warnings = validator.check_resources(config)
        if resource_warnings:
            if "warning" not in results:
                results["warning"] = []
            results["warning"].extend(resource_warnings)
        
        return print_validation_results(results, verbose)
        
    except Exception as e:
        print(f"Error validating current configuration: {e}")
        return 1


def compare_profiles(profile1: str, profile2: str) -> int:
    """Compare two configuration profiles."""
    print(f"Comparing profiles: {profile1} vs {profile2}")
    
    try:
        manager = get_config_manager()
        
        # Load first profile
        p1 = ConfigProfile(profile1)
        manager.load_profile(p1)
        config1 = manager.config.copy()
        
        # Load second profile
        p2 = ConfigProfile(profile2)
        manager.load_profile(p2)
        config2 = manager.config.copy()
        
        # Find differences
        differences = []
        
        def compare_dicts(d1: dict, d2: dict, path: str = ""):
            for key in set(d1.keys()) | set(d2.keys()):
                current_path = f"{path}.{key}" if path else key
                
                if key not in d1:
                    differences.append(f"  {current_path}: only in {profile2}")
                elif key not in d2:
                    differences.append(f"  {current_path}: only in {profile1}")
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    compare_dicts(d1[key], d2[key], current_path)
                elif d1[key] != d2[key]:
                    differences.append(f"  {current_path}: {d1[key]} → {d2[key]}")
        
        compare_dicts(config1, config2)
        
        if differences:
            print(f"\nDifferences found ({len(differences)}):")
            for diff in sorted(differences):
                print(diff)
        else:
            print("No differences found")
        
        return 0
        
    except Exception as e:
        print(f"Error comparing profiles: {e}")
        return 1


def list_profiles() -> int:
    """List all available configuration profiles."""
    print("Available configuration profiles:\n")
    
    from src.giflab.config_profiles import PROFILE_DESCRIPTIONS
    
    for name, description in PROFILE_DESCRIPTIONS.items():
        print(f"  {name:15} - {description}")
    
    return 0


def export_config(output_path: Path, profile: Optional[str] = None) -> int:
    """Export configuration to a file."""
    try:
        manager = get_config_manager()
        
        # Load profile if specified
        if profile:
            p = ConfigProfile(profile)
            manager.load_profile(p)
            print(f"Loaded profile: {profile}")
        
        # Export configuration
        config_data = manager.export_config()
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        print(f"Configuration exported to: {output_path}")
        return 0
        
    except Exception as e:
        print(f"Error exporting configuration: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate GifLab configuration files and profiles"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Validate file command
    file_parser = subparsers.add_parser("file", help="Validate a configuration file")
    file_parser.add_argument("path", type=Path, help="Path to configuration file")
    file_parser.add_argument("-v", "--verbose", action="store_true",
                            help="Show info messages")
    
    # Validate profile command
    profile_parser = subparsers.add_parser("profile", 
                                          help="Validate a configuration profile")
    profile_parser.add_argument("name", help="Profile name")
    profile_parser.add_argument("-v", "--verbose", action="store_true",
                               help="Show info messages")
    
    # Validate current command
    current_parser = subparsers.add_parser("current",
                                          help="Validate current configuration")
    current_parser.add_argument("-v", "--verbose", action="store_true",
                               help="Show info messages")
    
    # Compare profiles command
    compare_parser = subparsers.add_parser("compare",
                                          help="Compare two profiles")
    compare_parser.add_argument("profile1", help="First profile")
    compare_parser.add_argument("profile2", help="Second profile")
    
    # List profiles command
    subparsers.add_parser("list", help="List available profiles")
    
    # Export command
    export_parser = subparsers.add_parser("export",
                                         help="Export configuration to file")
    export_parser.add_argument("output", type=Path, help="Output file path")
    export_parser.add_argument("-p", "--profile", help="Profile to load before export")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == "file":
        return validate_file(args.path, args.verbose)
    elif args.command == "profile":
        return validate_profile(args.name, args.verbose)
    elif args.command == "current":
        return validate_current(args.verbose)
    elif args.command == "compare":
        return compare_profiles(args.profile1, args.profile2)
    elif args.command == "list":
        return list_profiles()
    elif args.command == "export":
        return export_config(args.output, args.profile)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())