import h5py
import os
import numpy as np
import importlib
import ast

from typing import Tuple, Any, Optional

from sdm.core.enc import TokenEncoder, CharacterPositionEncoder

# Type aliases to help Pylance
H5File = Any
H5Group = Any
H5Dataset = Any
H5Attrs = Any


def save_sdr(sdm: Any, encoder, filepath: str) -> None:
    """
    Save the sparse distributed representation memory system to disk using HDF5

    Args:
        sdm: Memory instance (TemporalMemory, TriadicTemporalMemory, etc.)
        encoder: Encoder instance (TokenEncoder, CharacterPositionEncoder, etc.)
        filepath: Path to save the system
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Remove file if it exists to ensure clean overwrite
    if os.path.exists(filepath):
        os.remove(filepath)

    with h5py.File(filepath, "w") as f:
        # Save memory attributes
        attrs = f.create_group("attributes")
        # Store memory type as string
        memory_type = f"{sdm.__class__.__module__}.{sdm.__class__.__name__}"
        attrs.attrs["memory_type"] = memory_type

        # Store encoder type as string
        encoder_type = f"{encoder.__class__.__module__}.{encoder.__class__.__name__}"
        attrs.attrs["encoder_type"] = encoder_type

        # Store dimension and population
        dimension = getattr(sdm, "dimension", getattr(sdm, "n", 0))
        population = getattr(sdm, "population", getattr(sdm, "p", 0))
        attrs.attrs["dimension"] = dimension
        attrs.attrs["population"] = population

        # Save memory matrices
        matrices = f.create_group("matrices")
        if hasattr(sdm, "M0") and hasattr(sdm.M0, "mem"):
            matrices.create_dataset("M0_mem", data=sdm.M0.mem)
        if hasattr(sdm, "M1") and hasattr(sdm.M1, "mem"):
            matrices.create_dataset("M1_mem", data=sdm.M1.mem)
        if hasattr(sdm, "M2") and hasattr(sdm.M2, "mem"):
            matrices.create_dataset("M2_mem", data=sdm.M2.mem)

        # Save encoder mappings
        enc_group = f.create_group("encoder")

        # Save tok2sdr (token -> SDR mapping)
        tok2sdr = enc_group.create_group("tok2sdr")

        # Replace problematic characters in token names
        char_replacements = {
            ".": "_DOT_",
            ",": "_COMMA_",
            "?": "_QMARK_",
            # '/': '_SLASH_',
            # '+': '_PLUS_',
            # '@': '_AT_',
            # '%': '_PERCENT_'
        }

        # Save tokens for TokenEncoder
        if hasattr(encoder, "tok2sdr"):
            for token, sdr in encoder.tok2sdr.items():
                # Create safe dataset name
                safe_name = str(token)
                for char, replacement in char_replacements.items():
                    safe_name = safe_name.replace(char, replacement)

                # Create dataset with safe name
                dset = tok2sdr.create_dataset(safe_name, data=sdr)
                # Store original token as attribute
                dset.attrs["original_token"] = str(token)

            # Save sdr2tok (SDR -> token mapping)
            sdr2tok = enc_group.create_group("sdr2tok")

            # Create a dataset to store all SDR tuples as integer arrays
            sdr_tuples_group = sdr2tok.create_group("sdr_tuples")

            # Store each SDR tuple as a dataset with its corresponding token
            for i, (sdr_tuple, token) in enumerate(encoder.sdr2tok.items()):
                # Convert tuple to numpy array of integers
                sdr_array = np.array(sdr_tuple, dtype=np.int32)

                # Create dataset with index-based name
                dset = sdr_tuples_group.create_dataset(f"sdr_{i}", data=sdr_array)

                # Store corresponding token
                dset.attrs["token"] = token

        # Save CharacterPositionEncoder specific attributes if it's that type
        if isinstance(encoder, CharacterPositionEncoder):
            char_pos_group = enc_group.create_group("char_position")

            # Save character to SDR mapping
            if hasattr(encoder, "char2sdr"):
                char2sdr_group = char_pos_group.create_group("char2sdr")
                for char, sdr in encoder.char2sdr.items():
                    # Use character code as name to avoid issues with special characters
                    safe_name = f"char_{ord(char)}"
                    dset = char2sdr_group.create_dataset(safe_name, data=sdr)
                    dset.attrs["character"] = char

            # Save position to SDR mapping
            if hasattr(encoder, "pos2sdr"):
                pos2sdr_group = char_pos_group.create_group("pos2sdr")
                for pos, sdr in encoder.pos2sdr.items():
                    pos_name = f"pos_{pos}"
                    dset = pos2sdr_group.create_dataset(pos_name, data=sdr)
                    dset.attrs["position"] = pos


def load_sdr(
    filepath: str,
    memory_class: Optional[Any] = None,
    encoder_class: Optional[Any] = None,
) -> Tuple[Any, Any]:
    """
    Load the sparse distributed representation memory system from disk

    Args:
        filepath: Path to the saved system
        memory_class: Optional class to use for memory initialization
        encoder_class: Optional class to use for encoder initialization

    Returns:
        Tuple[Any, Any]: Tuple of (memory, encoder) where memory is an instance
                         of the memory class and encoder is an instance of the
                         appropriate encoder class
    """
    with h5py.File(filepath, "r") as f:
        # Load attributes
        attrs: H5Group = f["attributes"]  # type: ignore
        dimension = attrs.attrs["dimension"]  # type: ignore
        population = attrs.attrs["population"]  # type: ignore

        # Import the correct memory class
        if memory_class is None:
            # Ensure memory_type is a string
            memory_type = str(attrs.attrs["memory_type"])  # type: ignore
            if "." in memory_type:
                module_name, class_name = memory_type.rsplit(".", 1)

                module = importlib.import_module(module_name)
                MemoryClass = getattr(module, class_name)
            else:
                raise ValueError(f"Invalid memory_type format: {memory_type}")
        else:
            MemoryClass = memory_class

        # Import the correct encoder class
        if encoder_class is None:
            # Check if encoder_type is stored
            if "encoder_type" in attrs.attrs:
                encoder_type = str(attrs.attrs["encoder_type"])  # type: ignore
                if "." in encoder_type:
                    module_name, class_name = encoder_type.rsplit(".", 1)

                    module = importlib.import_module(module_name)
                    EncoderClass = getattr(module, class_name)
                else:
                    raise ValueError(f"Invalid encoder_type format: {encoder_type}")
            else:
                # Default to TokenEncoder for backward compatibility
                EncoderClass = TokenEncoder
        else:
            EncoderClass = encoder_class

        # Create new instances
        memory = MemoryClass(dimension, population)

        # Initialize the appropriate encoder type
        if EncoderClass == CharacterPositionEncoder:
            # Initialize CharacterPositionEncoder with basic parameters
            encoder = EncoderClass(dimension, population)
        else:
            # Default for TokenEncoder and other types
            encoder = EncoderClass(dimension, population)

        # Restore memory matrices
        try:
            matrices_group: H5Group = f["matrices"]  # type: ignore

            # Check for each matrix and restore if it exists
            if hasattr(memory, "M0"):
                try:
                    dataset: H5Dataset = matrices_group["M0_mem"]  # type: ignore
                    memory.M0.mem = np.array(dataset)
                except KeyError:
                    pass

            if hasattr(memory, "M1"):
                try:
                    dataset: H5Dataset = matrices_group["M1_mem"]  # type: ignore
                    memory.M1.mem = np.array(dataset)
                except KeyError:
                    pass

            if hasattr(memory, "M2"):
                try:
                    dataset: H5Dataset = matrices_group["M2_mem"]  # type: ignore
                    memory.M2.mem = np.array(dataset)
                except KeyError:
                    pass
        except KeyError:
            print("Warning: No matrices found in the file")

        # Restore encoder-specific data
        try:
            # Get encoder group
            encoder_group: H5Group = f["encoder"]  # type: ignore

            # Handle specific encoder types
            if isinstance(encoder, TokenEncoder):
                # Initialize dictionaries
                encoder.tok2sdr = {}
                encoder.sdr2tok = {}

                # Restore token to SDR mappings
                try:
                    tok2sdr_group: H5Group = encoder_group["tok2sdr"]  # type: ignore

                    # Manual iteration through items
                    for key in tok2sdr_group:  # type: ignore
                        try:
                            dataset: H5Dataset = tok2sdr_group[key]  # type: ignore

                            # Check if original token is stored as attribute
                            if "original_token" in dataset.attrs:
                                token = dataset.attrs["original_token"]
                            else:
                                # Legacy format - key is the token
                                token = key

                            encoder.tok2sdr[token] = np.array(dataset)
                        except (KeyError, TypeError) as e:
                            print(f"Warning: Could not restore token {key}: {e}")
                except KeyError as e:
                    print(f"Warning: Could not restore tok2sdr mappings: {e}")

                # Restore SDR to token mappings
                try:
                    sdr2tok_group: H5Group = encoder_group["sdr2tok"]  # type: ignore

                    # Try the new format first (sdr_tuples group)
                    try:
                        sdr_tuples_group: H5Group = sdr2tok_group["sdr_tuples"]  # type: ignore

                        # Iterate through all SDR datasets
                        for key in sdr_tuples_group:  # type: ignore
                            try:
                                dataset: H5Dataset = sdr_tuples_group[key]  # type: ignore

                                if "token" in dataset.attrs:
                                    # Convert the array to a tuple of integers
                                    sdr_array = np.array(dataset)
                                    sdr_tuple = tuple(int(x) for x in sdr_array)

                                    # Get the associated token
                                    token = dataset.attrs["token"]

                                    # Store in the encoder
                                    encoder.sdr2tok[sdr_tuple] = token
                            except (KeyError, TypeError) as e:
                                print(f"Warning: Could not restore SDR {key}: {e}")

                    except KeyError:
                        # Fall back to old format (attributes on sdr2tok group)
                        print("Using legacy format for sdr2tok")
                        attrs: H5Attrs = sdr2tok_group.attrs  # type: ignore

                        # Get all attributes
                        for sdr_str in attrs:  # type: ignore
                            try:
                                # Handle both string representations:
                                # 1. Simple Python tuple representation
                                # 2. NumPy array representation

                                # Extract just the numbers from complex representations
                                # using regex to find all numbers
                                import re

                                numbers = re.findall(r"\d+", str(sdr_str))
                                if numbers:
                                    # Convert strings to integers and create a tuple
                                    sdr_tuple = tuple(int(num) for num in numbers)
                                    encoder.sdr2tok[sdr_tuple] = attrs[sdr_str]  # type: ignore
                                else:
                                    # Try ast.literal_eval as fallback for simpler strings
                                    try:
                                        sdr_tuple = ast.literal_eval(sdr_str)
                                        if isinstance(sdr_tuple, tuple):
                                            encoder.sdr2tok[sdr_tuple] = attrs[sdr_str]  # type: ignore
                                    except (SyntaxError, ValueError):
                                        print(
                                            f"Warning: Could not parse SDR string {sdr_str}"
                                        )
                            except Exception as e:
                                print(f"Warning: Error processing SDR {sdr_str}: {e}")
                except KeyError as e:
                    print(f"Warning: Could not restore sdr2tok mappings: {e}")

            # Handle CharacterPositionEncoder specific data
            elif isinstance(encoder, CharacterPositionEncoder):
                try:
                    char_pos_group = encoder_group["char_position"]

                    # Restore character to SDR mapping
                    if hasattr(encoder, "char2sdr") and "char2sdr" in char_pos_group:
                        char2sdr_group = char_pos_group["char2sdr"]

                        # Initialize with an empty dict if not already initialized
                        if not encoder.char2sdr:
                            encoder.char2sdr = {}

                        # Restore from datasets
                        for key in char2sdr_group:
                            dataset = char2sdr_group[key]
                            if "character" in dataset.attrs:
                                char = dataset.attrs["character"]
                                encoder.char2sdr[char] = np.array(dataset)

                    # Restore position to SDR mapping
                    if hasattr(encoder, "pos2sdr") and "pos2sdr" in char_pos_group:
                        pos2sdr_group = char_pos_group["pos2sdr"]

                        # Initialize with an empty dict if not already initialized
                        if not encoder.pos2sdr:
                            encoder.pos2sdr = {}

                        # Restore from datasets
                        for key in pos2sdr_group:
                            dataset = pos2sdr_group[key]
                            if "position" in dataset.attrs:
                                pos = int(dataset.attrs["position"])
                                encoder.pos2sdr[pos] = np.array(dataset)

                    # Restore token mappings (tok2sdr and sdr2tok should already be loaded from the common section)

                except KeyError as e:
                    print(
                        f"Warning: Could not restore CharacterPositionEncoder data: {e}"
                    )

        except KeyError as e:
            print(f"Warning: Could not restore encoder mappings: {e}")

        return memory, encoder


# Example usage:
"""
# Save trained system with TokenEncoder
save_sdr(memory, encoder, "models/temporal_memory_token_enc.h5")

# Save trained system with CharacterPositionEncoder
save_sdr(memory, char_pos_encoder, "models/temporal_memory_char_pos_enc.h5")

# Later, load the systems
from sdm.specialized_tm import SpecializedTemporalMemory  # import specific memory class
from sdm.enc import TokenEncoder, CharacterPositionEncoder  # import specific encoder classes

# Specify both memory and encoder classes
memory, encoder = load_sdr("models/temporal_memory_token_enc.h5", 
                          SpecializedTemporalMemory, 
                          TokenEncoder)

# Load with CharacterPositionEncoder
memory, char_pos_encoder = load_sdr("models/temporal_memory_char_pos_enc.h5",
                                    SpecializedTemporalMemory,
                                    CharacterPositionEncoder)

# Or let it auto-detect both classes (works for either encoder type)
memory, encoder = load_sdr("models/temporal_memory_token_enc.h5")
memory, char_pos_encoder = load_sdr("models/temporal_memory_char_pos_enc.h5")

# Generate text with loaded system
continuation = generate_sequence(memory, encoder, ["the", "quick"], 15)
print(" ".join(continuation))
"""
