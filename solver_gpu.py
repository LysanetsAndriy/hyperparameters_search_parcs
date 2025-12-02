from Pyro4 import expose
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger('PARCS_GridSearch')


def parallel_worker_call(workers, method_name, *common_args, per_worker_args=None, timeout=None):
    """
    Universal function for parallel method execution on PARCS workers
    
    Args:
        workers: list - List of Pyro4 worker proxies
        method_name: str - Name of the method to call on each worker
        *common_args: Arguments that are the same for all workers
        per_worker_args: list of tuples - Specific arguments for each worker (optional)
        timeout: float - Maximum time to wait for all workers (seconds)
    
    Returns:
        list: Results from workers in order of worker IDs
    """
    p = len(workers)
    
    if p == 0:
        raise ValueError("No workers provided")
    
    with ThreadPoolExecutor(max_workers=p) as executor:
        future_to_id = {}
        
        for i in range(p):
            worker_method = getattr(workers[i], method_name)
            
            if per_worker_args and i < len(per_worker_args):
                args = common_args + per_worker_args[i] + (i, p)
            else:
                args = common_args + (i, p)
            
            future = executor.submit(worker_method, *args)
            future_to_id[future] = i
        
        results = [None] * p
        
        for future in as_completed(future_to_id, timeout=timeout):
            worker_id = future_to_id[future]
            result = future.result()
            results[worker_id] = result
    
    return results


# ============================================================
# SOLVER CLASS
# ============================================================

class Solver:
    def __init__(self, workers=None, input_file=None, output_file=None):
        self.input_file = input_file
        self.output_file = output_file
        self.workers = workers if workers else []
        self.all_logs = []  # Collect all logs for output file
    
    def solve(self):
        """
        Master logic: Generate hyperparameter grid, distribute to workers, aggregate results
        """
        log.info("=" * 80)
        log.info("GRID SEARCH FOR MOBILENETV2 ON CIFAR-10")
        log.info("=" * 80)
        
        # Read configuration
        num_epochs, dataset_url = self.read_input()
        log.info(f"Configuration: {num_epochs} epochs")
        log.info(f"Dataset URL: {dataset_url}")
        
        # Generate hyperparameter grid
        hyperparameter_grid = self.generate_grid()
        total_configs = len(hyperparameter_grid)
        log.info(f"Total configurations to test: {total_configs}")
        
        # Distribute configurations to workers
        num_workers = len(self.workers)
        log.info(f"Available workers: {num_workers}")
        
        all_results = []
        
        # Process in batches (number of workers at a time)
        num_batches = (total_configs + num_workers - 1) // num_workers
        
        for batch_num in range(num_batches):
            batch_start = batch_num * num_workers
            batch_end = min(batch_start + num_workers, total_configs)
            batch_configs = hyperparameter_grid[batch_start:batch_end]
            
            log.info("-" * 80)
            log.info(f"Processing batch {batch_num + 1}/{num_batches}")
            log.info(f"Configurations {batch_start+1} to {batch_end} of {total_configs}")
            log.info("-" * 80)
            
            # Prepare per-worker arguments
            per_worker_args = [(config,) for config in batch_configs]
            
            # Execute in parallel
            try:
                batch_results = parallel_worker_call(
                    self.workers[:len(batch_configs)],  # Use only needed workers
                    "train_model",
                    num_epochs, dataset_url,  # Common args
                    per_worker_args=per_worker_args
                )
                
                all_results.extend(batch_results)
                
                # Log batch results
                for result in batch_results:
                    if result['status'] == 'success':
                        log.info(f"Config {result['config_id']}: "
                                 f"acc={result['best_val_acc']:.2f}%, "
                                 f"time={result['training_time']:.1f}s, "
                                 f"device={result['device']}") # ADDED DEVICE
                        # Collect worker logs
                        if 'logs' in result:
                            self.all_logs.extend(result['logs'])
                    else:
                        log.error(f"Config {result['config_id']}: FAILED - {result['error']}")
                
            except Exception as e:
                log.error(f"ERROR in batch {batch_num + 1}: {e}")
        
        log.info("=" * 80)
        log.info("GRID SEARCH COMPLETED")
        log.info("=" * 80)
        
        # Find and log best result
        successful = [r for r in all_results if r['status'] == 'success']
        if successful:
            best = max(successful, key=lambda x: x['best_val_acc'])
            log.info(f"Best configuration: ID={best['config_id']}, "
                     f"acc={best['best_val_acc']:.2f}%")
        
        # Write results
        self.write_output(all_results, total_configs)
        log.info(f"Results written to {self.output_file}")
    
    def generate_grid(self):
        """
        Generate all hyperparameter combinations
        
        Returns:
            list of dicts: Each dict contains one hyperparameter configuration
        """
        # Define hyperparameter space
        learning_rates = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03]
        batch_sizes = [32, 64]
        optimizers = ['sgd', 'adam']
        weight_decays = [0.0]
        fine_tuning_strategies = ['full'] #['full', 'partial']
        
        # Generate all combinations
        grid = []
        config_id = 0
        
        for lr, batch_size, optimizer, wd, fine_tune in itertools.product(
            learning_rates, batch_sizes, optimizers, weight_decays, fine_tuning_strategies
        ):
            config = {
                'config_id': config_id,
                'learning_rate': lr,
                'batch_size': batch_size,
                'optimizer': optimizer,
                'weight_decay': wd,
                'fine_tuning': fine_tune
            }
            grid.append(config)
            config_id += 1
        
        return grid
    
    @staticmethod
    @expose
    def train_model(num_epochs, dataset_url, config, worker_id, total_workers):
        """
        Worker logic: Train MobileNetV2 with specified hyperparameters
        
        Args:
            num_epochs: int - Number of training epochs
            dataset_url: str - GCS URL for CIFAR-10 dataset
            config: dict - Hyperparameter configuration
            worker_id: int - Worker identifier
            total_workers: int - Total number of workers
        
        Returns:
            dict: Training results and metrics
        """
        import time
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        import torchvision.models as models
        import numpy as np
        import urllib.request
        import os
        import logging
        
        # Worker-specific logger
        worker_log = logging.getLogger(f'Worker_{worker_id}')
        logs = []  # Collect logs to return to master
        
        start_time = time.time()
        
        try:
            worker_log.info(f"Starting training for config {config['config_id']}")
            worker_log.info(f"Hyperparameters: lr={config['learning_rate']}, "
                            f"batch={config['batch_size']}, "
                            f"opt={config['optimizer']}, "
                            f"finetune={config['fine_tuning']}")
            
            logs.append(f"[Worker {worker_id}] Config {config['config_id']} started")
            logs.append(f"[Worker {worker_id}] Params: {config}")
            
            
            # 1. Set device
            if torch.cuda.is_available():
                device = torch.device("cuda")
                device_name = torch.cuda.get_device_name(0)
                logs.append(f"[Worker {worker_id}] CUDA is available. Using device: {device_name}")
                worker_log.info(f"CUDA is available. Using device: {device_name}")
            else:
                device = torch.device("cpu")
                device_name = "cpu"
                logs.append(f"[Worker {worker_id}] CUDA not found. Using CPU.")
                worker_log.info("CUDA not found. Using CPU.")


            # Set random seed for reproducibility
            torch.manual_seed(42)
            np.random.seed(42)
            
            # Download dataset (with caching)
            dataset_path = f"/tmp/cifar10_worker_{worker_id}.npz"
            
            if not os.path.exists(dataset_path):
                worker_log.info("Downloading dataset...")
                logs.append(f"[Worker {worker_id}] Downloading dataset")
                urllib.request.urlretrieve(dataset_url, dataset_path)
                worker_log.info("Dataset downloaded")
                logs.append(f"[Worker {worker_id}] Dataset downloaded")
            else:
                worker_log.info("Using cached dataset")
                logs.append(f"[Worker {worker_id}] Using cached dataset")
            
            # Load dataset
            worker_log.info("Loading dataset...")
            data = np.load(dataset_path)
            
            X_train = data['X_train']  # (40000, 32, 32, 3) uint8
            y_train = data['y_train']  # (40000,)
            X_val = data['X_val']      # (10000, 32, 32, 3) uint8
            y_val = data['y_val']      # (10000,)
            mean = data['mean']        # (3,)
            std = data['std']          # (3,)
            
            TRAIN_SIZE = 800
            VAL_SIZE = 300

            X_train = X_train[:TRAIN_SIZE]
            y_train = y_train[:TRAIN_SIZE]
            X_val = X_val[:VAL_SIZE]
            y_val = y_val[:VAL_SIZE]

            worker_log.info(f"Using reduced dataset: {len(X_train)} train, {len(X_val)} val")

            
            # Normalize and convert to tensors
            X_train = torch.from_numpy(X_train).float().permute(0, 3, 1, 2) / 255.0
            X_val = torch.from_numpy(X_val).float().permute(0, 3, 1, 2) / 255.0
            
            # Normalize with ImageNet stats
            mean = torch.tensor(mean).view(1, 3, 1, 1)
            std = torch.tensor(std).view(1, 3, 1, 1)
            X_train = (X_train - mean) / std
            X_val = (X_val - mean) / std
            
            y_train = torch.from_numpy(y_train).long()
            y_val = torch.from_numpy(y_val).long()
            
            # Create DataLoaders
            batch_size = config['batch_size']
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            
            # Use pin_memory=True for faster data transfer to GPU
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
            
            worker_log.info(f"Data prepared: {len(train_dataset)} train, {len(val_dataset)} val")
            logs.append(f"[Worker {worker_id}] Data prepared: {len(train_dataset)} train samples")
            
            # Initialize model
            worker_log.info("Initializing MobileNetV2...")
            model = models.mobilenet_v2(pretrained=True)
            
            # Modify classifier for CIFAR-10 (10 classes)
            in_features = model.classifier[1].in_features # 1280

            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),

                nn.Linear(512, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),

                nn.Linear(256, 10)
            )
            
            # Apply fine-tuning strategy
            if config['fine_tuning'] == 'partial':
                # Freeze feature extractor, train only classifier
                for param in model.features.parameters():
                    param.requires_grad = False
                worker_log.info("Fine-tuning: Partial (classifier only)")
                logs.append(f"[Worker {worker_id}] Fine-tuning: Partial")
            else:
                # Train all layers
                worker_log.info("Fine-tuning: Full (all layers)")
                logs.append(f"[Worker {worker_id}] Fine-tuning: Full")
            
            # 2. Move model to device
            model.to(device)
            
            # 3. Move criterion to device
            criterion = nn.CrossEntropyLoss().to(device)

            if config['optimizer'] == 'sgd':
                optimizer = optim.SGD(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=config['learning_rate'],
                    momentum=0.9,
                    weight_decay=config['weight_decay']
                )
            else:  # adam
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=config['learning_rate'],
                    weight_decay=config['weight_decay']
                )
            
            # Training loop
            best_val_acc = 0.0
            train_losses = []
            val_accuracies = []
            
            worker_log.info(f"Starting training for {num_epochs} epochs...")
            logs.append(f"[Worker {worker_id}] Training started: {num_epochs} epochs")
            
            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    
                    # 4. Move data batch to device
                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                train_losses.append(avg_train_loss)
                
                # Validation phase
                model.eval()
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:

                        # 5. Move data batch to device
                        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                        
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                
                val_acc = 100.0 * correct / total
                val_accuracies.append(val_acc)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                epoch_log = (f"[Worker {worker_id}] Epoch {epoch+1}/{num_epochs}: "
                             f"Loss={avg_train_loss:.4f}, Val Acc={val_acc:.2f}%")
                worker_log.info(epoch_log)
                logs.append(epoch_log)
            
            training_time = time.time() - start_time
            
            final_log = (f"[Worker {worker_id}] Training completed in {training_time:.1f}s, "
                         f"Best Val Acc: {best_val_acc:.2f}%")
            worker_log.info(final_log)
            logs.append(final_log)
            
            # Return results
            return {
                'status': 'success',
                'config_id': config['config_id'],
                'config': config,
                'best_val_acc': best_val_acc,
                'final_train_loss': train_losses[-1],
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
                'training_time': training_time,
                'worker_id': worker_id,
                'logs': logs,
                'device': device_name # ADDED DEVICE NAME
            }
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            worker_log.error(f"ERROR: {e}")
            worker_log.error(error_msg)
            
            logs.append(f"[Worker {worker_id}] ERROR: {e}")
            
            return {
                'status': 'failed',
                'config_id': config.get('config_id', -1),
                'config': config,
                'error': str(e),
                'error_trace': error_msg,
                'worker_id': worker_id,
                'logs': logs
            }
    
    def read_input(self):
        """
        Read configuration from input file
        
        Returns:
            tuple: (num_epochs, dataset_url)
        """
        with open(self.input_file, 'r') as f:
            lines = f.readlines()
            num_epochs = int(lines[0].strip())
            dataset_url = lines[1].strip()
        
        return num_epochs, dataset_url

    def write_output(self, results, total_configs):
        """
        Write grid search results to output file

        Args:
            results: list of dicts - Results from all configurations
            total_configs: int - Total number of configurations
        """
        with open(self.output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("GRID SEARCH RESULTS: MobileNetV2 on CIFAR-10\n")
            f.write("=" * 80 + "\n\n")

            # Separate successful and failed results
            successful = [r for r in results if r['status'] == 'success']
            failed = [r for r in results if r['status'] == 'failed']

            f.write(f"Total Configurations: {total_configs}\n")
            f.write(f"Successful: {len(successful)}\n")
            f.write(f"Failed: {len(failed)}\n\n")

            if successful:
                # Sort by validation accuracy (descending)
                successful_sorted = sorted(successful, key=lambda x: x['best_val_acc'], reverse=True)

                # Best configuration
                best = successful_sorted[0]
                f.write("=" * 80 + "\n")
                f.write("BEST CONFIGURATION\n")
                f.write("=" * 80 + "\n")
                f.write(f"Config ID: {best['config_id']}\n")
                f.write(f"Learning Rate: {best['config']['learning_rate']}\n")
                f.write(f"Batch Size: {best['config']['batch_size']}\n")
                f.write(f"Optimizer: {best['config']['optimizer'].upper()}\n")
                f.write(f"Weight Decay: {best['config']['weight_decay']}\n")
                f.write(f"Best Validation Accuracy: {best['best_val_acc']:.2f}%\n")
                f.write(f"Final Training Loss: {best['final_train_loss']:.4f}\n")
                f.write(f"Training Time: {best['training_time']:.1f}s\n")
                f.write(f"Worker ID: {best['worker_id']}\n")
                f.write(f"Trained on: {best['device']}\n\n") # ADDED DEVICE

                # Compact table with all results
                f.write("=" * 80 + "\n")
                f.write("ALL RESULTS (Sorted by Validation Accuracy)\n")
                f.write("=" * 80 + "\n\n")

                # Table header
                f.write("Rank | Config |    LR     | Batch | Opt  |    WD     | Time(s) |  Loss  | Val Acc | Device\n")
                f.write("-----|--------|-----------|-------|------|-----------|---------|--------|---------|-------\n")

                # Table rows
                for rank, result in enumerate(successful_sorted, 1):
                    cfg = result['config']
                    device_str = "GPU" if "NVIDIA" in result['device'] else "cpu"
                    f.write(f"{rank:4d} | "
                            f"{result['config_id']:6d} | "
                            f"{cfg['learning_rate']:9.4f} | "
                            f"{cfg['batch_size']:5d} | "
                            f"{cfg['optimizer']:4s} | "
                            f"{cfg['weight_decay']:9.5f} | "
                            f"{result['training_time']:7.1f} | "
                            f"{result['final_train_loss']:6.4f} | "
                            f"{result['best_val_acc']:7.2f}% | "
                            f"{device_str}\n") # ADDED DEVICE

                f.write("\n")

            # Failed configurations
            if failed:
                f.write("=" * 80 + "\n")
                f.write("FAILED CONFIGURATIONS\n")
                f.write("=" * 80 + "\n\n")

                for result in failed:
                    cfg = result['config']
                    f.write(f"Config ID: {result['config_id']}\n")
                    f.write(f"  Hyperparameters: lr={cfg['learning_rate']}, "
                            f"batch={cfg['batch_size']}, "
                            f"opt={cfg['optimizer']}, "
                            f"wd={cfg.get('weight_decay', 0)}\n")
                    f.write(f"  Error: {result['error']}\n")
                    f.write(f"  Worker ID: {result['worker_id']}\n\n")

            # Append all worker logs
            f.write("=" * 80 + "\n")
            f.write("DETAILED WORKER LOGS\n")
            f.write("=" * 80 + "\n\n")

            for log_entry in self.all_logs:
                f.write(log_entry + "\n")

            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        log.info("Output file written successfully")
