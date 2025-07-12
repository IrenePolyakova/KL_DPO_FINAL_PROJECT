from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import datetime
import logging
from typing import Dict, List, Optional, Union
from pydantic import BaseModel
import hashlib
import uuid
from io import BytesIO

logger = logging.getLogger(__name__)

class TmxEntry(BaseModel):
    source_text: str
    target_text: str
    source_lang: str
    target_lang: str
    creation_date: datetime.datetime = datetime.datetime.now()
    last_modified: datetime.datetime = datetime.datetime.now()
    creator_tool: str = "Translation Evaluator"
    creator_id: Optional[str] = None
    custom_attributes: Dict[str, str] = {}
    metrics: Optional[Dict[str, float]] = None

class TmxCreator:
    def __init__(
        self,
        source_lang: str = "ru",
        target_lang: str = "en",
        creator_tool: str = "Translation Evaluator",
        creator_id: Optional[str] = None
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.creator_tool = creator_tool
        self.creator_id = creator_id or str(uuid.uuid4())
        
    def _create_header(self) -> Element:
        """Создает заголовок TMX с метаданными для MemoQ"""
        header = Element('header')
        header.set('creationtool', self.creator_tool)
        header.set('creationtoolversion', '1.0')
        header.set('segtype', 'sentence')
        header.set('o-tmf', 'MemoQ')
        header.set('adminlang', 'en')
        header.set('srclang', self.source_lang)
        header.set('datatype', 'PlainText')
        header.set('creationdate', datetime.datetime.now().strftime('%Y%m%dT%H%M%SZ'))
        header.set('creationid', self.creator_id)
        header.set('changedate', datetime.datetime.now().strftime('%Y%m%dT%H%M%SZ'))
        header.set('changeid', self.creator_id)
        
        # Добавляем prop для совместимости с MemoQ
        prop = SubElement(header, 'prop')
        prop.set('type', 'x-filename')
        prop.text = f'translation_memory_{datetime.datetime.now().strftime("%Y%m%d")}'
        
        return header
    
    def _create_tu_element(self, entry: TmxEntry) -> Element:
        """Создает элемент translation unit (tu) для TMX"""
        tu = Element('tu')
        
        # Генерируем уникальный tuid на основе текста
        content_hash = hashlib.md5(
            f"{entry.source_text}{entry.target_text}".encode()
        ).hexdigest()
        tu.set('tuid', content_hash)
        tu.set('creationdate', entry.creation_date.strftime('%Y%m%dT%H%M%SZ'))
        tu.set('creationid', self.creator_id)
        tu.set('changedate', entry.last_modified.strftime('%Y%m%dT%H%M%SZ'))
        tu.set('changeid', self.creator_id)
        
        # Добавляем пользовательские атрибуты
        for key, value in entry.custom_attributes.items():
            prop = SubElement(tu, 'prop')
            prop.set('type', key)
            prop.text = str(value)
        
        # Исходный текст
        tuv_src = SubElement(tu, 'tuv')
        tuv_src.set('xml:lang', entry.source_lang)
        seg_src = SubElement(tuv_src, 'seg')
        seg_src.text = entry.source_text
        
        # Перевод
        tuv_tgt = SubElement(tu, 'tuv')
        tuv_tgt.set('xml:lang', entry.target_lang)
        seg_tgt = SubElement(tuv_tgt, 'seg')
        seg_tgt.text = entry.target_text
        
        # Добавляем метрики в note, если они есть
        if entry.metrics:
            note = SubElement(tuv_tgt, 'note')
            metrics_str = ", ".join(
                f"{k}: {v:.2f}" for k, v in entry.metrics.items()
            )
            note.text = metrics_str
        
        return tu

    def create_tmx(self, entries: List[TmxEntry], output_file: str) -> str:
        """
        Создает TMX файл из списка записей
        
        Args:
            entries: список записей для включения в TMX
            output_file: путь для сохранения файла
            
        Returns:
            str: путь к созданному файлу
        """
        try:
            # Создаем корневой элемент
            tmx = Element('tmx', version='1.4')
            
            # Добавляем заголовок
            tmx.append(self._create_header())
            
            # Создаем body
            body = SubElement(tmx, 'body')
            
            # Добавляем записи
            for entry in entries:
                body.append(self._create_tu_element(entry))
            
            # Форматируем XML
            xml_str = minidom.parseString(tostring(tmx, encoding='unicode')).toprettyxml(indent="  ")
            
            # Сохраняем файл
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(xml_str)
            
            logger.info(f"TMX файл успешно создан: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Ошибка при создании TMX файла: {str(e)}")
            raise

    def create_tmx_string(self, entries: List[TmxEntry]) -> str:
        """
        Создает TMX в виде строки из списка записей
        
        Args:
            entries: список записей для включения в TMX
            
        Returns:
            str: TMX в виде строки
        """
        try:
            # Создаем корневой элемент
            tmx = Element('tmx', version='1.4')
            
            # Добавляем заголовок
            tmx.append(self._create_header())
            
            # Создаем body
            body = SubElement(tmx, 'body')
            
            # Добавляем записи
            for entry in entries:
                body.append(self._create_tu_element(entry))
            
            # Форматируем XML
            xml_str = minidom.parseString(tostring(tmx, encoding='unicode')).toprettyxml(indent="  ")
            
            return xml_str
            
        except Exception as e:
            logger.error(f"Ошибка при создании TMX строки: {str(e)}")
            raise

def create_tmx_file(
    src_sentences: List[str],
    tgt_sentences: List[str],
    fileobj: Optional[Union[str, 'BytesIO']] = None,
    sys_name: Optional[str] = None,
    source_lang: str = "ru",
    target_lang: str = "en",
    batch_size: int = 1000
) -> Union[str, bytes]:
    """
    Создает TMX файл с переводами
    
    Args:
        src_sentences: исходные предложения
        tgt_sentences: переведенные предложения
        fileobj: файловый объект или путь для сохранения
        sys_name: имя системы перевода
        source_lang: код исходного языка
        target_lang: код целевого языка
        batch_size: размер пакета для обработки
        
    Returns:
        Union[str, bytes]: путь к файлу или содержимое в виде bytes
    """
    try:
        creator = TmxCreator(source_lang=source_lang, target_lang=target_lang)
        entries = []
        
        # Создаем записи для переводов
        for src, tgt in zip(src_sentences, tgt_sentences):
            if not src.strip() or not tgt.strip():
                continue
                
            entries.append(TmxEntry(
                source_text=src,
                target_text=tgt,
                source_lang=source_lang,
                target_lang=target_lang,
                custom_attributes={'system': sys_name} if sys_name else {}
            ))
        
        # Если передан путь к файлу
        if isinstance(fileobj, str):
            return creator.create_tmx(entries, fileobj)
        
        # Если передан BytesIO или None
        xml_str = creator.create_tmx_string(entries)
        if fileobj is not None:
            fileobj.write(xml_str.encode('utf-8'))
            return fileobj
        
        return xml_str.encode('utf-8')
            
    except Exception as e:
        logger.error(f"Ошибка при создании TMX файла: {str(e)}")
        raise
