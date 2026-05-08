"""
数据脱敏工具模块
支持手机号、身份证、银行卡号等敏感信息脱敏
"""

import re
from typing import Pattern


class Desensitizer:
    """
    数据脱敏器
    对敏感信息进行脱敏处理
    """

    # 手机号正则（中国大陆）
    PHONE_PATTERN: Pattern[str] = re.compile(
        r'(?:^|\D)(1[3-9]\d{9})(?:$|\D)'
    )

    # 身份证号正则（18位）
    ID_CARD_PATTERN: Pattern[str] = re.compile(
        r'(?:^|\D)([1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx])(?:$|\D)'
    )

    # 银行卡号正则（16-19位）
    BANK_CARD_PATTERN: Pattern[str] = re.compile(
        r'(?:^|\D)(\d{16,19})(?:$|\D)'
    )

    # 邮箱正则
    EMAIL_PATTERN: Pattern[str] = re.compile(
        r'([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    )

    def __init__(
        self,
        phone_mask: str = "****",
        id_card_mask: str = "********",
        bank_card_mask: str = "************"
    ) -> None:
        """
        初始化脱敏器

        Args:
            phone_mask: 手机号脱敏掩码
            id_card_mask: 身份证号脱敏掩码
            bank_card_mask: 银行卡号脱敏掩码
        """
        self.phone_mask = phone_mask
        self.id_card_mask = id_card_mask
        self.bank_card_mask = bank_card_mask

    def desensitize_phone(self, phone: str) -> str:
        """
        脱敏手机号，保留前3后4位

        Args:
            phone: 手机号

        Returns:
            脱敏后的手机号
        """
        if not phone or len(phone) != 11:
            return phone
        return f"{phone[:3]}{self.phone_mask}{phone[-4:]}"

    def desensitize_id_card(self, id_card: str) -> str:
        """
        脱敏身份证号，保留前6后4位

        Args:
            id_card: 身份证号

        Returns:
            脱敏后的身份证号
        """
        if not id_card or len(id_card) != 18:
            return id_card
        return f"{id_card[:6]}{self.id_card_mask}{id_card[-4:]}"

    def desensitize_bank_card(self, bank_card: str) -> str:
        """
        脱敏银行卡号，保留前6后4位

        Args:
            bank_card: 银行卡号

        Returns:
            脱敏后的银行卡号
        """
        if not bank_card or len(bank_card) < 10:
            return bank_card
        return f"{bank_card[:6]}{self.bank_card_mask}{bank_card[-4:]}"

    def desensitize_email(self, email: str) -> str:
        """
        脱敏邮箱，保留域名，用户名脱敏

        Args:
            email: 邮箱地址

        Returns:
            脱敏后的邮箱
        """
        if not email or '@' not in email:
            return email

        username, domain = email.split('@', 1)
        if len(username) <= 2:
            masked_username = '*' * len(username)
        else:
            masked_username = f"{username[:2]}{'*' * (len(username) - 2)}"
        return f"{masked_username}@{domain}"

    def desensitize_text(
        self,
        text: str,
        desensitize_phone: bool = True,
        desensitize_id_card: bool = True,
        desensitize_bank_card: bool = True,
        desensitize_email: bool = True
    ) -> str:
        """
        脱敏文本中的所有敏感信息

        Args:
            text: 原始文本
            desensitize_phone: 是否脱敏手机号
            desensitize_id_card: 是否脱敏身份证号
            desensitize_bank_card: 是否脱敏银行卡号
            desensitize_email: 是否脱敏邮箱

        Returns:
            脱敏后的文本
        """
        if not text:
            return text

        result = text

        if desensitize_phone:
            result = self.PHONE_PATTERN.sub(
                lambda m: m.group(0)[:1] + self.desensitize_phone(m.group(1)) + m.group(0)[-1:],
                result
            )

        if desensitize_id_card:
            result = self.ID_CARD_PATTERN.sub(
                lambda m: m.group(0)[:1] + self.desensitize_id_card(m.group(1)) + m.group(0)[-1:],
                result
            )

        if desensitize_bank_card:
            result = self.BANK_CARD_PATTERN.sub(
                lambda m: m.group(0)[:1] + self.desensitize_bank_card(m.group(1)) + m.group(0)[-1:],
                result
            )

        if desensitize_email:
            result = self.EMAIL_PATTERN.sub(
                lambda m: self.desensitize_email(m.group(0)),
                result
            )

        return result

    def desensitize_dict(
        self,
        data: dict,
        keys_to_desensitize: list[str] | None = None
    ) -> dict:
        """
        脱敏字典中的敏感字段

        Args:
            data: 原始字典
            keys_to_desensitize: 需要脱敏的字段列表，默认自动检测敏感字段

        Returns:
            脱敏后的字典
        """
        if not data:
            return data

        result = data.copy()
        sensitive_keywords = [
            'phone', 'mobile', '电话', '手机',
            'id_card', 'idcard', '身份证',
            'bank_card', 'bankcard', 'card_no', 'cardno', '银行卡',
            'email', 'mail', '邮箱'
        ]

        for key, value in result.items():
            if isinstance(value, str):
                should_desensitize = False

                if keys_to_desensitize and key in keys_to_desensitize:
                    should_desensitize = True
                else:
                    for keyword in sensitive_keywords:
                        if keyword in key.lower():
                            should_desensitize = True
                            break

                if should_desensitize:
                    result[key] = self.desensitize_text(value)

        return result


# 创建全局脱敏器实例
default_desensitizer = Desensitizer()
