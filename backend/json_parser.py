import json
from typing import List, Dict, Any
from pathlib import Path


def parse_knowledge_json(json_path: str) -> List[Dict[str, Any]]:
    """
    JSON 파일에서 지식 데이터를 파싱하여 청크 형태로 변환
    
    Args:
        json_path: JSON 파일 경로
        
    Returns:
        [{"text": "내용", "index": 0, "metadata": {...}}, ...]
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {json_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파일 형식이 올바르지 않습니다: {e}")
    
    chunks = []
    chunk_index = 0
    
    # JSON 구조에 따라 다양한 형태 처리
    if isinstance(data, dict):
        # 단일 객체인 경우
        chunks.extend(_process_dict_object(data, chunk_index))
        chunk_index = len(chunks)
        
    elif isinstance(data, list):
        # 배열인 경우
        for item in data:
            if isinstance(item, dict):
                chunks.extend(_process_dict_object(item, chunk_index))
                chunk_index = len(chunks)
            elif isinstance(item, str):
                # 문자열 배열인 경우
                chunks.append({
                    "text": item,
                    "index": chunk_index,
                    "metadata": {"type": "text_item"}
                })
                chunk_index += 1
            else:
                # 기타 타입은 문자열로 변환
                chunks.append({
                    "text": str(item),
                    "index": chunk_index,
                    "metadata": {"type": "converted_item"}
                })
                chunk_index += 1
                
    else:
        # 기타 타입은 문자열로 변환
        chunks.append({
            "text": str(data),
            "index": 0,
            "metadata": {"type": "converted_data"}
        })
    
    return chunks


def _process_dict_object(obj: Dict[str, Any], start_index: int) -> List[Dict[str, Any]]:
    """
    딕셔너리 객체를 처리하여 청크로 변환
    
    Args:
        obj: 딕셔너리 객체
        start_index: 시작 인덱스
        
    Returns:
        청크 리스트
    """
    chunks = []
    current_index = start_index
    
    for key, value in obj.items():
        if isinstance(value, str):
            # 문자열 값인 경우
            if len(value.strip()) > 0:
                chunks.append({
                    "text": f"{key}: {value}",
                    "index": current_index,
                    "metadata": {
                        "type": "key_value",
                        "key": key,
                        "source": "json"
                    }
                })
                current_index += 1
                
        elif isinstance(value, dict):
            # 중첩된 딕셔너리인 경우
            nested_chunks = _process_dict_object(value, current_index)
            for chunk in nested_chunks:
                chunk["metadata"]["parent_key"] = key
                chunk["text"] = f"{key} > {chunk['text']}"
            chunks.extend(nested_chunks)
            current_index += len(nested_chunks)
            
        elif isinstance(value, list):
            # 배열 값인 경우
            for i, item in enumerate(value):
                if isinstance(item, str) and len(item.strip()) > 0:
                    chunks.append({
                        "text": f"{key}[{i}]: {item}",
                        "index": current_index,
                        "metadata": {
                            "type": "array_item",
                            "key": key,
                            "array_index": i,
                            "source": "json"
                        }
                    })
                    current_index += 1
                elif isinstance(item, dict):
                    # 배열 내 딕셔너리인 경우
                    nested_chunks = _process_dict_object(item, current_index)
                    for chunk in nested_chunks:
                        chunk["metadata"]["parent_key"] = key
                        chunk["metadata"]["array_index"] = i
                        chunk["text"] = f"{key}[{i}] > {chunk['text']}"
                    chunks.extend(nested_chunks)
                    current_index += len(nested_chunks)
                else:
                    # 기타 타입은 문자열로 변환
                    chunks.append({
                        "text": f"{key}[{i}]: {str(item)}",
                        "index": current_index,
                        "metadata": {
                            "type": "array_converted",
                            "key": key,
                            "array_index": i,
                            "source": "json"
                        }
                    })
                    current_index += 1
        else:
            # 기타 타입은 문자열로 변환
            chunks.append({
                "text": f"{key}: {str(value)}",
                "index": current_index,
                "metadata": {
                    "type": "converted_value",
                    "key": key,
                    "source": "json"
                }
            })
            current_index += 1
    
    return chunks


def validate_knowledge_json(json_path: str) -> bool:
    """
    JSON 파일의 유효성을 검증
    
    Args:
        json_path: JSON 파일 경로
        
    Returns:
        유효하면 True, 아니면 False
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def get_json_structure_info(json_path: str) -> Dict[str, Any]:
    """
    JSON 파일의 구조 정보를 반환
    
    Args:
        json_path: JSON 파일 경로
        
    Returns:
        구조 정보 딕셔너리
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        info = {
            "file_path": json_path,
            "file_size": Path(json_path).stat().st_size,
            "root_type": type(data).__name__,
            "total_items": 0,
            "structure": {}
        }
        
        if isinstance(data, dict):
            info["total_items"] = len(data)
            info["structure"] = _analyze_dict_structure(data)
        elif isinstance(data, list):
            info["total_items"] = len(data)
            info["structure"] = _analyze_list_structure(data)
        else:
            info["structure"] = {"type": type(data).__name__, "value": str(data)[:100]}
            
        return info
        
    except Exception as e:
        return {
            "file_path": json_path,
            "error": str(e),
            "valid": False
        }


def _analyze_dict_structure(obj: Dict[str, Any], max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
    """딕셔너리 구조 분석"""
    if current_depth >= max_depth:
        return {"type": "dict", "truncated": True}
    
    structure = {"type": "dict", "keys": {}}
    for key, value in obj.items():
        if isinstance(value, dict):
            structure["keys"][key] = _analyze_dict_structure(value, max_depth, current_depth + 1)
        elif isinstance(value, list):
            structure["keys"][key] = _analyze_list_structure(value, max_depth, current_depth + 1)
        else:
            structure["keys"][key] = {
                "type": type(value).__name__,
                "length": len(str(value)) if hasattr(value, '__len__') else 0
            }
    
    return structure


def _analyze_list_structure(obj: List[Any], max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
    """리스트 구조 분석"""
    if current_depth >= max_depth:
        return {"type": "list", "length": len(obj), "truncated": True}
    
    if not obj:
        return {"type": "list", "length": 0, "empty": True}
    
    # 첫 번째 요소의 타입을 기준으로 분석
    first_item = obj[0]
    structure = {
        "type": "list",
        "length": len(obj),
        "item_type": type(first_item).__name__
    }
    
    if isinstance(first_item, dict):
        structure["item_structure"] = _analyze_dict_structure(first_item, max_depth, current_depth + 1)
    elif isinstance(first_item, list):
        structure["item_structure"] = _analyze_list_structure(first_item, max_depth, current_depth + 1)
    
    return structure
