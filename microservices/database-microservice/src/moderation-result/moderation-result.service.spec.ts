import { Test, TestingModule } from '@nestjs/testing';
import { ModerationResultService } from './moderation-result.service';

describe('ModerationResultService', () => {
  let service: ModerationResultService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [ModerationResultService],
    }).compile();

    service = module.get<ModerationResultService>(ModerationResultService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });
});
