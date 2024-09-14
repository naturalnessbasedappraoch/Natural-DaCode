import os

def process_directory(directory):
    java_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root, file))
    return java_files

def read_file(file_path, encodings=('utf-8', 'iso-8859-1')):
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    raise Exception(f"Could not read the file {file_path} with provided encodings.")

def write_to_file(files, output_path):
    # Create the output directory if it does not exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, "w", encoding="utf-8") as output_file:
        for file_path in files:
            file_content = read_file(file_path)
            output_file.write("<s>\n")
            output_file.write(file_content)
            output_file.write("\n</s>\n")

def format_file(input_file_path, output_file_path):
    # Open the input file and the output file
    with open(input_file_path, 'r', encoding='utf-8') as input_file, \
         open(output_file_path, 'w', encoding='utf-8') as output_file:
        
        in_segment = False
        buffer = []

        for line in input_file:
            # Check for the start of <s> segment
            if '<s>' in line:
                in_segment = True
                # Append part of line after <s>
                buffer.append(line[line.index('<s>') + 3:])
                continue
            
            # Check for the end of </s> segment
            if '</s>' in line:
                # Append part of line before </s>
                buffer.append(line[:line.index('</s>')])
                # Process the entire buffered segment
                segment_content = ''.join(buffer)
                # Normalize spaces and newlines within the segment
                formatted_content = ' '.join(segment_content.split())
                # Write formatted content
                output_file.write('<s>' + formatted_content + '</s>\n')
                in_segment = False
                buffer = []
                # Write remainder of line after </s>
                output_file.write(line[line.index('</s>') + 4:])
                continue

            # If we're inside an <s>...</s> segment, buffer the line
            if in_segment:
                buffer.append(line)

if __name__ == "__main__":
    # Prompt the user for input and output paths
    input_directory = input("Enter the path to the folder containing .java files: ")
    output_file = input("Enter the path to the output .txt file: ")

    # Temporary file to store intermediate results
    javatextfile = "Train.txt"

    # Step 1: Convert .java files to one .txt file
    java_files = process_directory(input_directory)
    write_to_file(java_files, javatextfile)
    
    # Step 2: Format the generated .txt file to remove spaces
    format_file(javatextfile, output_file)
    
    # Optionally, remove the intermediate file
    os.remove(javatextfile)
