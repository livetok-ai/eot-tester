"""Command line interface for EOT Tester."""

import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table
from rich.text import Text

from .algorithms import TENAlgorithm, LiveKitAlgorithm, PipecatAlgorithm
from .base import EOTState


console = Console()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool) -> None:
    """EOT Tester - Test different end-of-turn detection algorithms."""
    if verbose:
        logging.basicConfig(level=logging.INFO)


@main.command()
@click.argument("text")
@click.option("--algorithm", "-a", required=True, help="Algorithm to use (ten, livekit-en, livekit-multilingual, pipecat)")
@click.option("--context", "-c", help="Optional context/system prompt")
def test(text: str, algorithm: str, context: Optional[str]) -> None:
    """Test EOT detection on given text."""
    
    # Initialize algorithm
    if algorithm.lower() == "ten":
        algo = TENAlgorithm()
    elif algorithm.lower() == "livekit-en":
        algo = LiveKitAlgorithm(model_type="en")
    elif algorithm.lower() == "livekit-multilingual":
        algo = LiveKitAlgorithm(model_type="multilingual")
    elif algorithm.lower() == "pipecat":
        algo = PipecatAlgorithm()
    else:
        console.print(f"[red]Unknown algorithm: {algorithm}[/red]")
        console.print("[yellow]Available algorithms: ten, livekit-en, livekit-multilingual, pipecat[/yellow]")
        return
    
    try:
        console.print(f"[blue]Initializing {algo.name} algorithm...[/blue]")
        algo.initialize()
        
        console.print(f"[blue]Testing text:[/blue] {text}")
        if context:
            console.print(f"[blue]Context:[/blue] {context}")
        
        result = algo.detect(text, context)
        
        # Display results
        table = Table(title="EOT Detection Result")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Algorithm", algo.name)
        table.add_row("State", result.state.value)
        table.add_row("Confidence", f"{result.confidence:.3f}")
        
        if result.metadata:
            for key, value in result.metadata.items():
                table.add_row(f"Meta: {key}", str(value))
        
        console.print(table)
        
        # Color-coded state display
        state_color = {
            EOTState.FINISHED: "green",
            EOTState.WAIT: "yellow", 
            EOTState.UNFINISHED: "red"
        }
        
        state_text = Text(
            f"Result: {result.state.value.upper()}", 
            style=f"bold {state_color[result.state]}"
        )
        console.print(state_text)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@main.command()
def interactive() -> None:
    """Interactive mode for testing multiple inputs."""
    console.print("[blue]Interactive EOT Testing Mode[/blue]")
    console.print("Type 'quit' to exit")
    
    algo = TENAlgorithm()
    try:
        console.print(f"[blue]Initializing {algo.name} algorithm...[/blue]")
        algo.initialize()
        
        while True:
            text = click.prompt("\nEnter text to test", type=str)
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            result = algo.detect(text)
            
            state_color = {
                EOTState.FINISHED: "green",
                EOTState.WAIT: "yellow",
                EOTState.UNFINISHED: "red"
            }
            
            console.print(f"State: [{state_color[result.state]}]{result.state.value}[/{state_color[result.state]}]")
            console.print(f"Confidence: {result.confidence:.3f}")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@main.command()
def list_algorithms() -> None:
    """List available EOT algorithms."""
    table = Table(title="Available EOT Algorithms")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="magenta")
    
    table.add_row("ten", "TEN Turn Detection using Qwen2.5-7B transformer model")
    table.add_row("livekit-en", "LiveKit English-only turn detector with ONNX optimization")
    table.add_row("livekit-multilingual", "LiveKit multilingual turn detector with contextual awareness")
    table.add_row("pipecat", "Pipecat Smart Turn V2 using audio classification on TTS-generated speech")
    
    console.print(table)


@main.command()
@click.option("--algorithm", "-a", required=True, help="Algorithm to use (ten, livekit-en, livekit-multilingual, pipecat)")
@click.option("--dataset-dir", "-d", default="dataset", help="Directory containing test files")
def evaluate(algorithm: str, dataset_dir: str) -> None:
    """Evaluate algorithm accuracy on test dataset."""
    
    # Initialize algorithm
    if algorithm.lower() == "ten":
        algo = TENAlgorithm()
    elif algorithm.lower() == "livekit-en":
        algo = LiveKitAlgorithm(model_type="en")
    elif algorithm.lower() == "livekit-multilingual":
        algo = LiveKitAlgorithm(model_type="multilingual")
    elif algorithm.lower() == "pipecat":
        algo = PipecatAlgorithm()
    else:
        console.print(f"[red]Unknown algorithm: {algorithm}[/red]")
        console.print("[yellow]Available algorithms: ten, livekit-en, livekit-multilingual, pipecat[/yellow]")
        return
    
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        console.print(f"[red]Dataset directory not found: {dataset_dir}[/red]")
        return
    
    # Test files and their expected states
    test_files = {
        "finished.txt": EOTState.FINISHED,
        "unfinished.txt": EOTState.UNFINISHED #,
        # "wait.txt": EOTState.WAIT
    }
    
    try:
        console.print(f"[blue]Initializing {algo.name} algorithm...[/blue]")
        algo.initialize()
        
        # Create results table
        results_table = Table(title="Evaluation Results")
        results_table.add_column("File", style="cyan")
        results_table.add_column("Total", style="white")
        results_table.add_column("Correct", style="green")
        results_table.add_column("Accuracy", style="magenta")
        
        overall_correct = 0
        overall_total = 0
        
        # Count total sentences for progress tracking
        file_sentence_counts = {}
        for filename, expected_state in test_files.items():
            file_path = dataset_path / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    sentences = [line.strip() for line in f if line.strip()]
                    file_sentence_counts[filename] = len(sentences)
        
        total_sentences = sum(file_sentence_counts.values())
        
        # with Progress() as progress:
        #     overall_task = progress.add_task("[cyan]Evaluating overall...", total=total_sentences)
            
        for filename, expected_state in test_files.items():
            file_path = dataset_path / filename
            
            if not file_path.exists():
                console.print(f"[yellow]Warning: {filename} not found, skipping...[/yellow]")
                continue
            
            # Read test sentences
            with open(file_path, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f if line.strip()]
            
            if not sentences:
                console.print(f"[yellow]Warning: {filename} is empty, skipping...[/yellow]")
                continue
            
            correct = 0
            total = len(sentences)
              
            # file_task = progress.add_task(f"[blue]Testing {filename}...", total=total)
            
            for sentence in sentences:
                try:
                    result = algo.detect(sentence)
                    if result.state == expected_state:
                        correct += 1
                except Exception as e:
                    console.print(f"[red]Error processing sentence: {e}[/red]")
                    continue
                
            #     progress.update(file_task, advance=1)
            #     progress.update(overall_task, advance=1)
            
            # progress.remove_task(file_task)
            
            accuracy = (correct / total) * 100 if total > 0 else 0
            results_table.add_row(
                filename,
                str(total),
                str(correct),
                f"{accuracy:.1f}%"
            )
            
            overall_correct += correct
            overall_total += total
        
        # Add overall results
        if overall_total > 0:
            overall_accuracy = (overall_correct / overall_total) * 100
            results_table.add_row(
                "[bold]OVERALL[/bold]",
                f"[bold]{overall_total}[/bold]",
                f"[bold]{overall_correct}[/bold]",
                f"[bold]{overall_accuracy:.1f}%[/bold]"
            )
        
        console.print(results_table)
        
    except Exception as e:
        console.print(f"[red]Error during evaluation: {e}[/red]")


if __name__ == "__main__":
    main()