import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { ClassificationResult } from './entities/classification-results.entity';

@Injectable()
export class ImageRetrievalService {
  constructor(
    @InjectRepository(ClassificationResult)
    private readonly classificationResultRepository: Repository<ClassificationResult>,
  ) {}

  async getImageByTaskId(taskId: string) {
    const result = await this.classificationResultRepository.findOne({
      where: { taskId },
      relations: ['labelProbabilities'], // Fetch the associated label probabilities
    });

    if (!result) {
      throw new NotFoundException('Image not found');
    }

    // Return the image result here
    return result;
  }
}
