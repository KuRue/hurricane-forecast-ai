"""Command-line interface for Hurricane Forecast AI."""

import sys
from pathlib import Path
from typing import Optional

import click
import torch
from loguru import logger
from rich.console import Console
from rich.table import Table

from .data.loaders import HURDAT2Loader, HurricaneDataPipeline
from .data.validators import HurricaneDataValidator
from .utils import setup_logging, get_config, log_gpu_info

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="Hurricane Forecast AI")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool):
    """Hurricane Forecast AI - Advanced hurricane prediction using deep learning."""
    ctx.ensure_object(dict)
    
    # Setup logging
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level=level)
    
    # Load configuration
    if config:
        ctx.obj["config"] = get_config(config_path=config)
    else:
        ctx.obj["config"] = get_config()
    
    # Welcome message
    if verbose:
        console.print("[bold cyan]Hurricane Forecast AI[/bold cyan]")
        console.print("Advanced hurricane prediction using deep learning\n")


@cli.command()
@click.option(
    "--data-only",
    is_flag=True,
    help="Download only data (no models)",
)
@click.option(
    "--models-only",
    is_flag=True,
    help="Download only models (no data)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force re-download of existing files",
)
def setup(data_only: bool, models_only: bool, force: bool):
    """Set up data and models for hurricane forecasting."""
    from scripts.setup_data import main as setup_main
    
    console.print("[bold]Setting up Hurricane Forecast AI...[/bold]\n")
    
    # Build arguments
    args = ["setup"]
    if not models_only:
        args.append("--download-era5")
    if not data_only:
        args.append("--download-models")
    if force:
        args.append("--force")
    
    # Run setup
    sys.argv = args
    result = setup_main()
    
    if result == 0:
        console.print("\n[bold green]✓ Setup completed successfully![/bold green]")
    else:
        console.print("\n[bold red]✗ Setup failed. Check the logs.[/bold red]")
        sys.exit(1)


@cli.command()
@click.argument("storm_id")
@click.option(
    "--source",
    type=click.Choice(["hurdat2", "ibtracs"]),
    default="hurdat2",
    help="Data source to use",
)
@click.option(
    "--validate",
    is_flag=True,
    help="Validate the storm data",
)
def info(storm_id: str, source: str, validate: bool):
    """Display information about a specific storm."""
    console.print(f"\n[bold]Storm Information: {storm_id}[/bold]\n")
    
    # Load storm data
    try:
        if source == "hurdat2":
            loader = HURDAT2Loader()
            loader.load()
            storm_track = loader.get_storm(storm_id)
        else:
            console.print("[red]IBTrACS support not yet implemented[/red]")
            return
        
        # Display basic info
        table = Table(title=f"Storm {storm_id} Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Track Points", str(len(storm_track)))
        table.add_row("Start Time", str(storm_track["timestamp"].min()))
        table.add_row("End Time", str(storm_track["timestamp"].max()))
        table.add_row("Duration", str(storm_track["timestamp"].max() - storm_track["timestamp"].min()))
        table.add_row("Peak Intensity", f"{storm_track['max_wind'].max():.0f} knots")
        table.add_row("Minimum Pressure", f"{storm_track['min_pressure'].min():.0f} mb")
        
        console.print(table)
        
        # Validate if requested
        if validate:
            console.print("\n[bold]Validating storm data...[/bold]")
            validator = HurricaneDataValidator()
            results = validator.validate_track(storm_track)
            
            if results["valid"]:
                console.print("[green]✓ Data validation passed[/green]")
            else:
                console.print("[red]✗ Data validation failed[/red]")
                for error in results["errors"]:
                    console.print(f"  [red]Error: {error}[/red]")
            
            if results["warnings"]:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in results["warnings"]:
                    console.print(f"  [yellow]Warning: {warning}[/yellow]")
                    
    except Exception as e:
        console.print(f"[red]Error loading storm {storm_id}: {e}[/red]")
        sys.exit(1)


@cli.command()
def gpu_status():
    """Check GPU availability and status."""
    console.print("\n[bold]GPU Status[/bold]\n")
    
    if torch.cuda.is_available():
        console.print(f"[green]✓ CUDA is available[/green]")
        console.print(f"CUDA Version: {torch.version.cuda}")
        console.print(f"Number of GPUs: {torch.cuda.device_count()}\n")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            
            table = Table(title=f"GPU {i}: {props.name}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Compute Capability", f"{props.major}.{props.minor}")
            table.add_row("Total Memory", f"{props.total_memory / 1024**3:.1f} GB")
            table.add_row("Memory Clock", f"{props.memory_clock_rate / 1000:.1f} MHz")
            table.add_row("Memory Bus Width", f"{props.memory_bus_width} bits")
            table.add_row("Multiprocessors", str(props.multi_processor_count))
            table.add_row("CUDA Cores", str(props.multi_processor_count * 64))  # Approximate
            
            console.print(table)
            
            # Current memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                console.print(f"\nMemory Usage:")
                console.print(f"  Allocated: {allocated:.2f} GB")
                console.print(f"  Reserved: {reserved:.2f} GB")
                console.print(f"  Free: {(props.total_memory / 1024**3) - reserved:.2f} GB\n")
    else:
        console.print("[red]✗ CUDA is not available[/red]")
        console.print("Running on CPU only")
        
        # Check CPU info
        import psutil
        console.print(f"\nCPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
        console.print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")


@cli.command()
@click.option(
    "--years",
    "-y",
    multiple=True,
    type=int,
    help="Years to include in statistics",
)
@click.option(
    "--basin",
    type=click.Choice(["atlantic", "pacific", "all"]),
    default="atlantic",
    help="Basin to analyze",
)
def stats(years: tuple, basin: str):
    """Display hurricane statistics."""
    console.print("\n[bold]Hurricane Statistics[/bold]\n")
    
    # Load data
    loader = HURDAT2Loader()
    storms_df, tracks_df = loader.load()
    
    # Filter by years if specified
    if years:
        storms_df = storms_df[storms_df["year"].isin(years)]
        storm_ids = storms_df["storm_id"].unique()
        tracks_df = tracks_df[tracks_df["storm_id"].isin(storm_ids)]
        console.print(f"Filtered to years: {sorted(years)}\n")
    
    # Calculate statistics
    total_storms = len(storms_df)
    hurricanes = tracks_df[tracks_df["max_wind"] >= 64]["storm_id"].nunique()
    major_hurricanes = tracks_df[tracks_df["max_wind"] >= 96]["storm_id"].nunique()
    
    # Display summary
    table = Table(title="Hurricane Statistics Summary")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Percentage", style="yellow")
    
    table.add_row("Total Storms", str(total_storms), "100.0%")
    table.add_row("Hurricanes (Cat 1+)", str(hurricanes), f"{hurricanes/total_storms*100:.1f}%")
    table.add_row("Major Hurricanes (Cat 3+)", str(major_hurricanes), f"{major_hurricanes/total_storms*100:.1f}%")
    
    console.print(table)
    
    # Yearly breakdown if not too many years
    if len(years) <= 10 or not years:
        console.print("\n[bold]Yearly Breakdown[/bold]")
        
        yearly = storms_df.groupby("year").size()
        yearly_hurricanes = tracks_df[tracks_df["max_wind"] >= 64].groupby(
            tracks_df["storm_id"].str[4:8].astype(int)
        )["storm_id"].nunique()
        
        year_table = Table()
        year_table.add_column("Year", style="cyan")
        year_table.add_column("Total Storms", style="green")
        year_table.add_column("Hurricanes", style="yellow")
        
        for year in sorted(yearly.index)[-10:]:  # Last 10 years
            total = yearly.get(year, 0)
            hurr = yearly_hurricanes.get(year, 0)
            year_table.add_row(str(year), str(total), str(hurr))
        
        console.print(year_table)


@cli.command()
@click.argument("storm_id")
@click.option(
    "--hours",
    type=int,
    default=120,
    help="Forecast horizon in hours",
)
@click.option(
    "--ensemble-size",
    type=int,
    default=50,
    help="Number of ensemble members",
)
@click.option(
    "--model",
    type=click.Choice(["graphcast", "pangu", "ensemble"]),
    default="ensemble",
    help="Model to use for forecasting",
)
def forecast(storm_id: str, hours: int, ensemble_size: int, model: str):
    """Generate a forecast for a hurricane (placeholder)."""
    console.print(f"\n[bold]Generating forecast for {storm_id}[/bold]\n")
    console.print("[yellow]Note: This is a placeholder. Full forecasting will be implemented in Phase 2.[/yellow]\n")
    
    console.print(f"Configuration:")
    console.print(f"  Model: {model}")
    console.print(f"  Forecast hours: {hours}")
    console.print(f"  Ensemble members: {ensemble_size}")
    
    console.print("\n[cyan]To implement:[/cyan]")
    console.print("  1. Load the specified model")
    console.print("  2. Prepare input data for the storm")
    console.print("  3. Run ensemble forecast")
    console.print("  4. Generate forecast products")
    console.print("  5. Save results and visualizations")


@cli.command()
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Port to run the API server",
)
@click.option(
    "--host",
    type=str,
    default="0.0.0.0",
    help="Host to bind the API server",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development",
)
def serve(port: int, host: str, reload: bool):
    """Start the API server (placeholder)."""
    console.print(f"\n[bold]Starting API Server[/bold]\n")
    console.print("[yellow]Note: This is a placeholder. API server will be implemented in Phase 4.[/yellow]\n")
    
    console.print(f"Configuration:")
    console.print(f"  Host: {host}")
    console.print(f"  Port: {port}")
    console.print(f"  Auto-reload: {reload}")
    
    console.print("\n[cyan]API will include:[/cyan]")
    console.print("  POST /forecast - Generate hurricane forecast")
    console.print("  GET /storms - List available storms")
    console.print("  GET /storms/{id} - Get storm details")
    console.print("  GET /health - Health check endpoint")
    console.print("  GET /docs - Interactive API documentation")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
