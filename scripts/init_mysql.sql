-- ==============================================================================
-- RAG_XPER Enterprise - MySQL Database Initialization Script
-- Database: rag_xper_db (UTF-8 Multi-byte support for Arabic)
-- ==============================================================================

CREATE DATABASE IF NOT EXISTS `rag_xper_db`
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE `rag_xper_db`;

-- ------------------------------------------------------------------------------
-- Table: books (Documents & Books Catalog)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `books` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `title` VARCHAR(255) NOT NULL,
    `author` VARCHAR(255) NULL,
    `category` VARCHAR(100) DEFAULT 'General',
    `filename` VARCHAR(255) NOT NULL UNIQUE,
    `file_path` VARCHAR(500) NOT NULL,
    `file_size_bytes` BIGINT DEFAULT 0,
    `content_hash` VARCHAR(64) NULL,
    `total_pages` INT DEFAULT 1,
    `chunk_count` INT DEFAULT 0,
    `strategy_used` VARCHAR(50) DEFAULT 'recursive',
    `status` VARCHAR(50) DEFAULT 'indexed',
    `notes` TEXT NULL,
    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX `idx_title` (`title`),
    INDEX `idx_category` (`category`),
    INDEX `idx_content_hash` (`content_hash`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------------------------
-- Table: query_logs (Chat History & Telemetry)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `query_logs` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `session_id` VARCHAR(100) NULL,
    `question` TEXT NOT NULL,
    `answer` TEXT NOT NULL,
    `reasoning` TEXT NULL,
    `sources_json` LONGTEXT NULL,
    `execution_time_ms` FLOAT DEFAULT 0.0,
    `is_cached` BOOLEAN DEFAULT FALSE,
    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX `idx_session_id` (`session_id`),
    INDEX `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
