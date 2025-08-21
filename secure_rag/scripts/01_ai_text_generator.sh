#!/bin/bash

SOURCE_DIR="$HOME/secure_rag/current/"
OUTPUT_DIR="$HOME/secure_rag/md"
LOG_FILE="$OUTPUT_DIR/processing.log"

# Создаем выходной каталог и лог-файл
mkdir -p "$OUTPUT_DIR"
> "$LOG_FILE"

# Функция для обработки HTML
process_html() {
    local file="$1"
    local relative_path="${file#$SOURCE_DIR/}"
    local output_path="$OUTPUT_DIR/${relative_path%.*}.clean.md"
    local output_dir=$(dirname "$output_path")
    
    mkdir -p "$output_dir"
    
    # Извлекаем title страницы
    local title=$(perl -0777 -ne 'print $1 if /<title>(.*?)<\/title>/i' "$file" | tr -d '\n')
    
    # Очищаем HTML
    perl -0777 -pe '
        s/<head>.*?<\/head>//gsi;
        s/<script\b[^>]*>.*?<\/script>//gs;
        s/<style\b[^>]*>.*?<\/style>//gs;
        s/<!--.*?-->//gs;
        s/<[^>]+>//g;
        s/\s+/ /g;
        s/^\s+|\s+$//g;
        s/(\S)\s+(\S)/$1 $2/g;
        s/&(nbsp|lt|gt|amp|quot|apos);/ /g;
    ' "$file" | awk 'NF {print}' > "$output_path"
    
    # Добавляем метаданные
    echo -e "# $title\n" | cat - "$output_path" > temp && mv temp "$output_path"
    echo -e "\n\n**Source:** \`$SOURCE_DIR/${relative_path}\`" >> "$output_path"
    echo "Processed HTML: $file -> $output_path" >> "$LOG_FILE"
}

# Функция для обработки Markdown
process_markdown() {
    local file="$1"
    local relative_path="${file#$SOURCE_DIR/}"
    local output_path="$OUTPUT_DIR/$relative_path"
    local output_dir=$(dirname "$output_path")
    
    mkdir -p "$output_dir"
    
    # Очищаем Markdown
    perl -0777 -pe '
        s/^---.*?---//gs;
        s/^#+\s*(.*?)\s*#*$/\n# $1\n/gsm;
        s/\[([^\]]+)\]\([^)]+\)/$1/gs;
        s/!\[([^\]]*)\]\([^)]+\)//gs;
        s/`{3}.*?`{3}//gs;
        s/`[^`]+`//gs;
        s/\*{1,2}([^*]+)\*{1,2}/$1/gs;
        s/_{1,2}([^_]+)_{1,2}/$1/gs;
        s/^\s*[-*+]\s+//gm;
        s/^\s*\d+\.\s+//gm;
        s/^\|.*\|$//gm;
        s/^>\s*//gm;
        s/\s+/ /g;
        s/^\s+|\s+$//g;
    ' "$file" | awk 'NF {print}' > "$output_path"
    
    # Добавляем ссылку на источник
    echo -e "\n\n**Source:** \`$SOURCE_DIR/${relative_path}\`" >> "$output_path"
    echo "Processed Markdown: $file -> $output_path" >> "$LOG_FILE"
}

# Основной цикл обработки
process_file() {
    local file="$1"
    case "${file##*.}" in
        html|htm) process_html "$file" ;;
        md) process_markdown "$file" ;;
        *) echo "Skipped unsupported file: $file" >> "$LOG_FILE" ;;
    esac
}

export -f process_html process_markdown process_file
export SOURCE_DIR OUTPUT_DIR LOG_FILE

# Обрабатываем все файлы с сохранением структуры каталогов
find "$SOURCE_DIR" -type f \( -iname "*.html" -o -iname "*.htm" -o -iname "*.md" \) \
    -exec bash -c 'process_file "$0"' {} \;

echo "Processing complete. Results saved to $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
